import torch
from app.milvus import get_collection
from app.model import MNISTEncoder

model = MNISTEncoder()
model.load_state_dict(torch.load("mnist_encoder.pt", map_location="cpu"))
model.eval()

collection = get_collection()

def search(image_tensor, top_k=5):
    with torch.no_grad():
        emb, _ = model(image_tensor.unsqueeze(0))

    results = collection.search(
        data=[emb.numpy()[0]],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["label"]
    )

    return results[0]
