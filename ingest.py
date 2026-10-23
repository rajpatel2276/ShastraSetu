import os
import uuid
import pymupdf
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import shutil

# Import shared client to prevent lock errors
from retrieve import client, collection_name

print("Initializing Multimodal Encoders...")
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

def embed_content(text=None, image=None):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        if image is not None:
            outputs = model.get_image_features(**inputs)
        else:
            outputs = model.get_text_features(**inputs)
            
    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        tensor = outputs.text_embeds
    elif hasattr(outputs, "image_embeds") and outputs.image_embeds is not None:
        tensor = outputs.image_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        tensor = outputs.pooler_output
    else:
        tensor = outputs[0]
        
    return [float(x) for x in tensor.detach().cpu().flatten().tolist()]

def ingest_pdf(pdf_path: str):
    """Dynamically called by the API. Clears old data, extracts, embeds, and loads Qdrant."""
    print(f"Processing Uploaded PDF: {pdf_path}")
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )

    img_dir = "data/images"
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    os.makedirs(img_dir, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    points = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # 1. Extract Text
        text = page.get_text()
        if text.strip():
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                vector = embed_content(text=chunk)
                points.append(
                    PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"type": "text", "content": chunk, "page": page_num + 1})
                )
                
        # 2. Extract Images (With strict dimension filtering)
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            
            # --- NEW JUNK FILTER ---
            img_width = base_image.get("width", 0)
            img_height = base_image.get("height", 0)
            
            # Ignore anything smaller than 250x250 (logos, icons, spacer lines, watermarks)
            if img_width < 250 or img_height < 250:
                continue 
            # -----------------------

            image_path = f"{img_dir}/page_{page_num + 1}_img_{img_index}.{base_image['ext']}"
            with open(image_path, "wb") as f:
                f.write(base_image["image"])
                
            pil_img = Image.open(image_path).convert("RGB")
            vector = embed_content(image=pil_img)
            
            points.append(
                PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"type": "image", "path": image_path, "page": page_num + 1})
            )

    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    return len(points)