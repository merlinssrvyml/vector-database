import torch
from app.milvus import get_collection
from app.model import MNISTEncoder
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------
# Configuration
# ----------------------------
EMBEDDING_DIM = 128 # Dimension of the embedding vector
BATCH_SIZE = 64
EPOCHS = 5
MODEL_PATH = "mnist_encoder.pt"

# ----------------------------
# Dataset
# ----------------------------
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,  # IMPORTANT: set True only if network allows
    transform=transform
)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----------------------------
# Model & Training Setup
# ----------------------------
model = MNISTEncoder(embedding_dim=EMBEDDING_DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

# ----------------------------
# Train
# ----------------------------
model.train()
for epoch in range(EPOCHS):
    total_loss = 0.0

    for images, labels in loader:
        optimizer.zero_grad()
        _, logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{EPOCHS} - loss: {total_loss:.4f}")

# ----------------------------
# Save trained model
# ----------------------------
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ----------------------------
# Insert embeddings into Milvus
# ----------------------------
collection = get_collection()

embeddings = []
labels = []

model.eval()
with torch.no_grad():
    for images, y in loader:
        emb, _ = model(images)

        # emb is already L2-normalized in the model
        embeddings.extend(emb.cpu().numpy())
        labels.extend(y.cpu().numpy())

collection.insert([embeddings, labels])
collection.flush()

print("Vectors inserted into Milvus successfully")
