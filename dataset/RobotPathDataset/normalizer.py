import torch
import numpy as np
class MinMaxFeatureNormalizer:
    def __init__(self):
        pass
    def initialize(self, min_values, max_values, device='cuda'):
        """
        Initializes the Normalizer with minimum and maximum values for each feature.

        Args:
            min_values (Tensor or list): The minimum values for each feature.
            max_values (Tensor or list): The maximum values for each feature.
        """
        self.device = device
        if isinstance(min_values, list):
            min_values = torch.tensor(min_values)
        if isinstance(max_values, list):
            max_values = torch.tensor(max_values)
        if isinstance(min_values, np.ndarray):
            min_values = torch.tensor(min_values)
        if isinstance(max_values, np.ndarray):
            max_values = torch.tensor(max_values)

        # Ensure the max values are greater than the min values
        assert torch.all(max_values > min_values), "Max values must be greater than min values"

        # move to device
        min_values = min_values.to(device)
        max_values = max_values.to(device)
        

        self.min_values = min_values
        self.max_values = max_values
        self.epsilon = 1e-8  # To avoid division by zero

    def normalize(self, x):
        """
        Normalize the input tensor using min-max normalization.

        Args:
            x (Tensor): The tensor to normalize with shape (batch, timesteps, features).
        
        Returns:
            Tensor: The normalized tensor, with values typically in the range [0, 1].
        """
        if self.min_values.device != x.device:
            self.min_values = self.min_values.to(x.device)
            self.max_values = self.max_values.to(x.device)
        # Normalize using min-max scaling
        x = (x - self.min_values) / (self.max_values - self.min_values + self.epsilon)
        return x  # ADD * 2 - 1  TO Normalize to the range [-1, 1]
    def denormalize(self, x, is_numpy=False):
        """
        Denormalize the input tensor using min-max normalization.

        Args:
            x (Tensor): The tensor to denormalize with shape (batch, timesteps, features).
        
        Returns:
            Tensor: The denormalized tensor.
        """
       
        if is_numpy:
            x = torch.tensor(x).to(self.device)
        # x = torch.clamp(x, -1, 1)
        x = torch.clamp(x, 0, 1)
        # Denormalize from [-1, 1] to original range
        # x = (x + 1) / 2


        value = x * (self.max_values - self.min_values) + self.min_values
        if is_numpy:
            return value.cpu().detach().numpy()
        else:
            return value

    def __call__(self, x):
        """
        Make the class callable for a more intuitive API.
        """
        return self.normalize(x)

if __name__=='__main__':
    # Example usage
    min_vals = [2.0, 5.0, 7.0]  # Minimum values for each feature
    max_vals = [10.0, 20.0, 30.0]  # Maximum values for each feature
    normalizer = MinMaxFeatureNormalizer(min_vals, max_vals)

    # Create a tensor with the shape (batch, timesteps, features)
    x = torch.tensor([
        [[5.0, 10.0, 15.0], [10.0, 20.0, 30.0]],  # Batch 1
        [[2.5, 5.0, 7.5], [7.5, 15.0, 22.5]]      # Batch 2
    ])

    print('x.shape:', x.shape)

    # Normalize the tensor along the features axis (last axis)
    normalized_x = normalizer(x)

    print("Original Tensor:")
    print(x)

    print("\nNormalized Tensor:")
    print(normalized_x)
