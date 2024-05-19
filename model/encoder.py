import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def forward(self, x):
        raise NotImplementedError("This method should be overridden by child class")

class WorldIndexEmbeder(Encoder):
    """
    A class representing a world index embedder.

    Args:
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The dimension of the embeddings.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Attributes:
        embedding (nn.Embedding): The embedding layer.
        device (str): The device used for computation.

    """

    def __init__(self, num_embeddings, embedding_dim, device='cuda'):
        super(WorldIndexEmbeder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        Forward pass of the world index embedder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The embedded tensor.

        """
        return self.embedding(x)


class FlattenEmbeder(Encoder):
    """
    A class representing a flatten embedder.

    Args:
        input_dim (int): The dimension of the input.
        embedding_dim (int): The dimension of the embeddings.
        device (str, optional): The device to use for computation. Defaults to 'cuda'.

    Attributes:
        embedding (nn.Linear): The embedding layer.
        device (str): The device used for computation.

    """

    def __init__(self, input_sample,  embedding_dim=64,  normalizer=None,device='cuda'):
        super(FlattenEmbeder, self).__init__()


        # get the input dimension
        input_dim = input_sample.view(-1).size(0)

        # flatten the input sample
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, embedding_dim),
            nn.SELU()
        )
        self.device = device
        self.to(device)

    def forward(self, x):
        """
        Forward pass of the flatten embedder.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The embedded tensor.

        """

        return self.embedding(x)
