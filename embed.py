import os
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

print("Loading CLIP Model...")
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

client = QdrantClient(path="./qdrant_data")
collection_name = "debate_evidence"

# Clean setup without deprecation warnings
if client.collection_exists(collection_name):
    client.delete_collection(collection_name)

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

def force_flat_list(output):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "text_embeds") and output.text_embeds is not None:
        tensor = output.text_embeds
    elif hasattr(output, "image_embeds") and output.image_embeds is not None:
        tensor = output.image_embeds
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    else:
        tensor = output[0]
    return [float(x) for x in tensor.detach().cpu().flatten().tolist()]

def get_text_embedding(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return force_flat_list(outputs)

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return force_flat_list(outputs)

def index_data(texts, images):
    print("Generating embeddings and uploading to Qdrant...")
    points = []
    point_id = 1
    
    # Process Texts
    for item in texts:
        content = item.get("text", item.get("content", "")) if isinstance(item, dict) else str(item)
        if not content.strip():
            continue
            
        vector = get_text_embedding(content)
        page = item.get("page", 0) if isinstance(item, dict) else 0
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={"type": "text", "content": content, "page": page}
        ))
        point_id += 1
            
    # Process Images
    for item in images:
        path = item.get("path", item.get("image", "")) if isinstance(item, dict) else str(item)
        if not path or not os.path.exists(path):
            continue
            
        vector = get_image_embedding(path)
        page = item.get("page", 0) if isinstance(item, dict) else 0
        points.append(PointStruct(
            id=point_id,
            vector=vector,
            payload={"type": "image", "path": path, "page": page}
        ))
        point_id += 1

    if points:
        client.upsert(collection_name=collection_name, points=points)
        print(f"Successfully indexed {len(points)} items into Qdrant.")
    else:
        print("No valid items found to index.")

if __name__ == "__main__":
    loaded_texts = []
    loaded_images = []
    
    # Load texts if stored in data/
    for text_path in ["data/texts.json", "data/chunks.json", "data/extracted_text.json"]:
        if os.path.exists(text_path):
            with open(text_path, "r") as f:
                loaded_texts = json.load(f)
            break
            
    # Load images
    img_dir = "data/images"
    if os.path.exists(img_dir):
        for img_name in sorted(os.listdir(img_dir)):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                loaded_images.append({"path": os.path.join(img_dir, img_name)})
                
    index_data(loaded_texts, loaded_images)
    
    # Explicit close to prevent destructor shutdown error
    client.close()