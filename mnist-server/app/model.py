import torch.nn as nn
import torch.nn.functional as F


class MNISTEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.embedding = nn.Linear(256, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        emb = self.embedding(x)
        emb = F.normalize(emb, p=2, dim=1)
        
        logits = self.classifier(emb)
        return emb, logits
