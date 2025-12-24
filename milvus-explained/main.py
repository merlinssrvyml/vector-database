import numpy as np
from pymilvus import (Collection, CollectionSchema, DataType, FieldSchema,
                      connections)

# Step 1: Connect to Milvus server
connections.connect("default", host="localhost", port="19530")

# Step 2: Define the schema for the collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Auto ID for primary key
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=100)  # 100-dimensional vectors
]

schema = CollectionSchema(fields, description="Vector database for AI applications")

# Step 3: Create a collection in Milvus
collection = Collection(name="vectors_collection", schema=schema)

# Step 4: Generate random vectors (100 vectors, each of 100 dimensions)
num_vectors = 100
dimension = 100
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Step 5: Insert the vectors into the collection
data = [vectors.tolist()]

# Insert the vectors into the collection
collection.insert(data)
print(f"Inserted {num_vectors} vectors into Milvus!")

# Step 6: Create an index for the 'vector' field
index_params = {
    "index_type": "IVF_FLAT",  # Index type, inverted file index for fast search
    "metric_type": "L2",  # Metric type, L2 for Euclidean distance
    "params": { # Additional parameters like nlist
        "nlist": 128, # controls how many partitions are used in the index. Larger values may result in faster search but require more memory
    }
}

# Create index on the 'vector' field
collection.create_index(field_name="vector", index_params=index_params)
print("Index created for the collection!")

# Step 7: Load the collection into memory before searching
collection.load()
print("Collection loaded into memory!")

# Step 8: Create a query vector (let's take the first vector as the query)
query_vector = vectors[0].reshape(1, -1)  # Query with the first vector

# Step 9: Perform a similarity search (find 3 nearest neighbors)
# nprobe is a parameter controlling how many partitions to search in the index, higher values give more accurate results but take longer
search_params = {"nprobe": 10}  # Set search parameters
results = collection.search(query_vector, "vector", search_params, limit=3)

# Step 10: Print the search results
print("\nSearch Results:")
for result in results[0]:
    print(f"ID: {result.id}, Distance: {result.distance}")

# Step 11: Optionally, drop the collection when done (clean up)
collection.drop()
