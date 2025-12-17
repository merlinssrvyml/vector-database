import numpy as np

# Step 1: Create random vectors (for learning purposes)
def generate_random_vectors(num_vectors, dimensions):
    vectors = np.random.rand(num_vectors, dimensions)
    return vectors

# Step 2: Cosine similarity function
def cosine_similarity(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

# Step 3: Find most similar vector (nearest neighbor)
def find_most_similar_vector(similarity_scores):
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores[0]  # Return the most similar vector

# Parameters
num_vectors = 5  # Number of vectors to generate
dimensions = 10  # Number of dimensions for each vector

# Generate random vectors
vectors = generate_random_vectors(num_vectors, dimensions)
print("Generated Vectors:")
print(vectors)

# Choose a query vector (let’s take the first one for simplicity)
query_vector = vectors[0]

# Calculate cosine similarities between the query and all other vectors
similarities = []
for i, vector in enumerate(vectors):
    similarity = cosine_similarity(query_vector, vector)
    similarities.append((i, similarity))

# Print similarity results
print("\nCosine Similarities between Query Vector and Other Vectors:")
for idx, similarity in similarities:
    print(f"Vector {idx}: Similarity = {similarity:.4f}")

# Find and print the most similar vector
most_similar = find_most_similar_vector(similarities)
print(f"\nThe most similar vector to the query vector is Vector {most_similar[0]} with similarity {most_similar[1]:.4f}")
