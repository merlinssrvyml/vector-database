import csv
from pathlib import Path

import numpy as np
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections, utility)
from sentence_transformers import SentenceTransformer

# -----------------------------
# 1. Connect to Milvus
# -----------------------------

connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

# -----------------------------
# 2. Load embedding model
# -----------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIM = model.get_sentence_embedding_dimension()

# -----------------------------
# 3. Define collection schema
# -----------------------------

fields = [
    FieldSchema(
        name="id",
        dtype=DataType.INT64,
        is_primary=True,
        auto_id=True
    ),
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=512
    ),
    FieldSchema(
        name="vector",
        dtype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM
    )
]

schema = CollectionSchema(
    fields=fields,
    description="Sentence semantic search"
)

collection_name = "sentence_search"



if utility.has_collection(collection_name):
    Collection(collection_name).drop()

collection = Collection(
    name=collection_name,
    schema=schema
)

# -----------------------------
# 4. Load sentences from CSV
# -----------------------------

csv_path = Path(__file__).parent / "sentences.csv"

sentences = []
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sentences.append(row["text"])

print(f"Loaded {len(sentences)} sentences from CSV.")

# -----------------------------
# 5. Generate embeddings
# -----------------------------

embeddings = model.encode(
    sentences,
    normalize_embeddings=True
).astype("float32")

# -----------------------------
# 6. Insert into Milvus
# -----------------------------

collection.insert([
    sentences,
    embeddings.tolist()
])

collection.flush()
print("Data inserted into Milvus.")

# -----------------------------
# 7. Create index
# -----------------------------

index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "IP",  # cosine similarity
    "params": {"nlist": 128}
}

collection.create_index(
    field_name="vector",
    index_params=index_params
)

collection.load()
print("Collection indexed and loaded.")

# -----------------------------
# 8. Query interesting phrases
# -----------------------------

queries = [
    "semantic meaning of text",
    "scalable backend systems",
    "machine learning data pipelines",
    "high availability in distributed systems",
    "vector search for AI applications"
]

search_params = {"nprobe": 10}

for query in queries:
    print(f"\nQUERY: {query}")

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    results = collection.search(
        data=query_embedding,
        anns_field="vector",
        param=search_params,
        limit=3,
        output_fields=["text"]
    )

    for rank, hit in enumerate(results[0], start=1):
        print(
            f"{rank}. score={hit.distance:.4f} | {hit.entity.get('text')}"
        )
