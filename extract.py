import os
import json
import shutil
import pymupdf 

def clean_workspace():
    print("Cleaning up old trash and database...")
    
    # 1. Delete the old Qdrant database to prevent ghost entries
    if os.path.exists("qdrant_data"):
        shutil.rmtree("qdrant_data")
        
    # 2. Delete and recreate the images folder
    img_dir = os.path.join("data", "images")
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)
        
    # 3. Delete old JSON files if they exist
    json_path = os.path.join("data", "texts.json")
    if os.path.exists(json_path):
        os.remove(json_path)

def extract_data():
    # Auto-detect the PDF in the data directory
    pdf_files = [f for f in os.listdir("data") if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("Error: No PDF found in the 'data' folder.")
        return
        
    pdf_path = os.path.join("data", pdf_files[0])
    print(f"Found PDF: {pdf_path}")
    
    doc = pymupdf.open(pdf_path)
    texts = []
    image_count = 0

    print("Extracting text and images...")
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # --- EXTRACT TEXT ---
        text = page.get_text("text").strip()
        if text:
            # Split into logical chunks by paragraph spacing
            chunks = text.split("\n\n")
            for chunk in chunks:
                clean_chunk = chunk.strip().replace("\n", " ")
                # Ignore tiny artifacts or page numbers (less than 30 chars)
                if len(clean_chunk) > 30:
                    texts.append({
                        "page": page_num + 1,
                        "text": clean_chunk 
                    })

        # --- EXTRACT IMAGES ---
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
            image_path = os.path.join("data", "images", image_name)
            
            with open(image_path, "wb") as f:
                f.write(image_bytes)
            image_count += 1

    # Save all text chunks directly to the JSON file embed.py is looking for
    with open(os.path.join("data", "texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, indent=4)

    print(f"Extraction complete! Saved {len(texts)} text chunks and {image_count} images.")

if __name__ == "__main__":
    clean_workspace()
    extract_data()