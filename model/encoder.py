import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This method should be overridden by child class")

class WorldIndexEmbeder(Encoder):
    def __init__(self, num_embeddings, embedding_dim, device='cuda'):
        super(WorldIndexEmbeder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        return self.embedding(x)
