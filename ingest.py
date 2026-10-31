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
    """
    First-principles semantic chunking.
    Eliminates mid-sentence PDF formatting breaks and respects logical sentence boundaries.
    """
    # 1. Heal artificial PDF line breaks within sentences, preserving true double-newline paragraphs
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 2. Split strictly by sentence boundaries (punctuation followed by whitespace)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding the next sentence stays under the limit, append it
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk += (" " + sentence) if current_chunk else sentence
        else:
            # If the chunk is full, lock it in and start a new one
            if current_chunk:
                chunks.append(current_chunk)
            # Retain the previous sentence as overlap to preserve context across boundaries
            current_chunk = sentence
            
    if current_chunk:
        chunks.append(current_chunk)
        
    return chunks

def ingest_pdf(pdf_path: str):
    print(f"Processing Uploaded PDF: {pdf_path}")
    start_time = time.time() 
    
    os.makedirs("data/pdfs", exist_ok=True)
    permanent_pdf_path = "data/pdfs/active_document.pdf"
    
    shutil.copy(pdf_path, permanent_pdf_path)
    
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
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
            # Apply the custom semantic chunker instead of the LangChain abstraction
            chunks = semantic_chunk_text(text, max_chars=800)
            
            for chunk in chunks:
                vector = embed_query(chunk) # Ensure this calls your 512-dim embedding function
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
    print(f"✅ Semantic Ingestion Complete: {len(points)} precise vectors embedded in {end_time - start_time:.2f} seconds.")
    return len(points)
