import torch.nn as nn
import torch
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

    def __init__(self, input_sample,  embedding_dim=64,  normalizer=None,device='cuda', num_of_hidden_layers=1):
        super(FlattenEmbeder, self).__init__()

        self.min_values = min_values = torch.ones((64, 64)) * -1
        self.max_values = max_values = torch.ones((64, 64)) * 3.5

        if normalizer is not None:
            self.normalizer = normalizer
            normalizer.initialize(min_values, max_values, device=device)
        else:
            self.normalizer = None

        # get the input dimension
        input_dim = input_sample.view(-1).size(0)
        
        # create hidden layers
        hidden_layers = []
        current_dim = input_dim
        for i in range(num_of_hidden_layers-1):
            hidden_layers.append(nn.Linear(current_dim, current_dim//2))
            hidden_layers.append(nn.SELU())
            current_dim = current_dim//2

        # flatten the input sample
        self.embedding = nn.Sequential(
            nn.Flatten(),
            # add the hidden layers
            *hidden_layers,
            nn.Linear(current_dim, embedding_dim),
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
        
        if self.normalizer is not None:
            x = self.normalizer(x)
        return self.embedding(x)


class VAEEncoder(Encoder):
    def __init__(self, vae_model):
        super(VAEEncoder, self).__init__()
        self.vae_model = vae_model
    def forward(self, x):
        results = self.vae_model.encode(x)
        mu, log_var = results
        z = self.vae_model.reparameterize(mu, log_var)
        return z

    def decode(self, z):
        return self.vae_model.decode(z)
