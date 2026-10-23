import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
client = QdrantClient(path="./qdrant_data")
collection_name = "debate_evidence"

def force_flat_list(output):
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "text_embeds") and output.text_embeds is not None:
        tensor = output.text_embeds
    elif hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensor = output.pooler_output
    else:
        tensor = output[0]
    return [float(x) for x in tensor.detach().cpu().flatten().tolist()]

def embed_query(text_query):
    inputs = processor(text=[text_query], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    return force_flat_list(outputs)

def search_evidence(query, limit=3, item_type=None):
    """Searches Qdrant with optional payload filtering (e.g., item_type='image') for high efficiency."""
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