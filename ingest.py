import os
import uuid
import pymupdf
import torch
import re
from transformers import CLIPProcessor, CLIPModel
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
import shutil
import time

# Import shared client to prevent lock errors
from retrieve import client, collection_name
from retrieve import embed_query

import time # Add this import
print("Initializing Multimodal Encoders...")
model_id = "openai/clip-vit-base-patch32"

# Detect RTX 3050 (CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device) # Move model to GPU

def embed_content(text=None, image=None):
    inputs = processor(text=text, images=image, return_tensors="pt", padding=True, truncation=True)
    # Move all input tensors to the GPU
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    
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

def semantic_chunk_text(text, max_chars=800):
    """First-principles semantic chunking respecting sentence boundaries."""
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def manage_pdf_storage(upload_dir="data/pdfs", max_files=5):
    """Enforces a strict FIFO limit on stored PDFs to prevent system bloat."""
    os.makedirs(upload_dir, exist_ok=True)
    files = [os.path.join(upload_dir, f) for f in os.listdir(upload_dir) if f.endswith('.pdf')]
    
    # If we exceed our maximum allowed files, delete the oldest ones
    if len(files) >= max_files:
        # Sort by last modification time (oldest first)
        files.sort(key=os.path.getmtime)
        files_to_delete = files[:len(files) - max_files + 1]
        
        for old_file in files_to_delete:
            try:
                os.remove(old_file)
                print(f"🗑️ Storage Manager: Purged old PDF to prevent bloat -> {old_file}")
            except Exception as e:
                print(f"⚠️ Failed to delete old file {old_file}: {e}")

def ingest_pdf(pdf_path: str, original_filename: str = "document.pdf"):
    print(f"Processing Uploaded PDF: {original_filename}")
    start_time = time.time() 
    
    safe_filename = original_filename.replace(' ', '_')
    permanent_pdf_path = os.path.join("data/pdfs", safe_filename)
    
    # 1. Check if this exact file is already in the knowledge base
    if os.path.exists(permanent_pdf_path):
        print(f"⚠️ File '{original_filename}' already exists in the knowledge base. Skipping duplicate ingestion.")
        return 0 # Return 0 vectors added since it's already there
        
    # 2. Otherwise save normally
    shutil.copy(pdf_path, permanent_pdf_path)
    
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    
    doc = pymupdf.open(permanent_pdf_path)
    points = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if text.strip():
            chunks = semantic_chunk_text(text, max_chars=800)
            for chunk in chunks:
                vector = embed_query(chunk)
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector=vector, 
                        payload={
                            "type": "text", 
                            "content": chunk, 
                            "page": page_num + 1,
                            "source_file": permanent_pdf_path
                        }
                    )
                )

    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    doc.close() 
    
    end_time = time.time()
    print(f"✅ Multi-Doc Ingestion Complete: {len(points)} vectors embedded for {original_filename} in {end_time - start_time:.2f} seconds.")
    return len(points)