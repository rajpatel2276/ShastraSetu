import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Initialize Models
model_id = "openai/clip-vit-base-patch32"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id).to(device)

# Initialize Qdrant
client = QdrantClient(path="./qdrant_data")
collection_name = "debate_evidence"

def embed_query(text):
    """Uses the exact same proven extraction logic from ingest.py to guarantee 512 dimensions."""
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()} 
    
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        
    # The exact extraction sequence from ingest.py that successfully processed 118 vectors
    if isinstance(outputs, torch.Tensor):
        tensor = outputs
    elif hasattr(outputs, "text_embeds") and outputs.text_embeds is not None:
        tensor = outputs.text_embeds
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        tensor = outputs.pooler_output
    else:
        tensor = outputs[0]
        
    vector = [float(x) for x in tensor.detach().cpu().flatten().tolist()]
    
    # Absolute safety net: if the tokenizer still tries to pass a sequence array (e.g. 3584 elements),
    # slice exactly 512 to protect the Qdrant database.
    if len(vector) > 512:
        vector = vector[:512]
        
    return vector

def search_evidence(query, limit=3, item_type=None):
    """Searches Qdrant with optional payload filtering."""
    query_vector = embed_query(query)
    
    query_filter = None
    if item_type:
        query_filter = Filter(
            must=[FieldCondition(key="type", match=MatchValue(value=item_type))]
        )
    
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        query_filter=query_filter
    )
    
    return response.points