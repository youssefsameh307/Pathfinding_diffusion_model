import torch
import torch.nn as nn

# torch create custom loss function
class ObstacleFreePathLoss(nn.Module):
    """
    A module that computes the loss for an obstacle-free path. It penelizes paths that are too close to obstacles. 
    the min_safe_distance param is used to define the minimum distance that is considered safe. 
    The loss is computed by interpolating the distance field values at the points of the path and summing them up. 
    The loss is negated to make it a minimization problem.
    
    The best acheivable value is = the the min_safe_distance. as any distance above that is considered safe and won't contribute

    Args:
        normalizer (optional): A normalization function to be applied to the input (default: None).
        min_safe_distance (float): The minimum safe distance to consider (default: 0.05).
        device (str): The device to run the computation on (default: 'cuda').
    """
    def __init__(self, normalizer=None, min_safe_distance=0.05, device='cuda'):
        super(ObstacleFreePathLoss, self).__init__()
        self.device = device    
        self.min_safe_distance = torch.tensor(min_safe_distance, device=device)
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = None
        
    def world_to_grid(self, world_coords, grid_size=64, world_size=10.0):
        """
        Convert continuous world coordinates to discrete grid coordinates.
        
        Parameters:
            world_coords (torch.Tensor): A 2D tensor of shape (B ,N, 2) representing the world coordinates (x, y).
            grid_size (int): Size of the grid (default is 64 for a 64x64 grid).
            world_size (float): Size of the world (default is 10.0 for a 10x10 world).
        
        Returns:
            torch.Tensor: A 2D tensor of shape (B ,N, 2) with the corresponding grid coordinates.
        """
        # Scale world coordinates to the grid
        scale = grid_size / world_size
        grid_coords = torch.clamp(world_coords * scale, 0, grid_size - 1)

        return grid_coords
        
    def forward(self, path, world_distance_field_img):
        """
        Compute the loss for an obstacle-free path.

        Parameters:
            path (torch.Tensor): A tensor of shape (batch, n_waypoints, 2) representing the path coordinates.
            world_distance_field_img (torch.Tensor): A tensor of shape (batch, 64, 64) representing the distance field.

        Returns:
            torch.Tensor: The computed loss value.
        """
        # path is a tensor of shape (batch,n_waypoints, 2)
        # world_distance_field_img is a tensor of shape (batch, 64, 64)
        # 1st interpolate the values of the distance field at the points of the path
        assert path.shape[2] == 2
        batch_size = path.shape[0]
        
        grid_size_x = world_distance_field_img.shape[1] 
        grid_size_y = world_distance_field_img.shape[2]
        
        
        if self.normalizer:
            path = self.normalizer.denormalize(path)
        # map the path to the grid
        path = self.world_to_grid(path, grid_size=64, world_size=10.0)
        # Get the integer and fractional parts of the points
        x = path[:, :, 0]
        y = path[:, :, 1]
        
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1
        
        # Clip values to be within the grid bounds
        x0 = torch.clamp(x0, 0, grid_size_x - 1)
        x1 = torch.clamp(x1, 0, grid_size_x - 1)
        y0 = torch.clamp(y0, 0, grid_size_y - 1)
        y1 = torch.clamp(y1, 0, grid_size_y - 1)

        # Get the values at the corners of the interpolation square
        Ia = world_distance_field_img[:, :,x0, y0]
        Ib = world_distance_field_img[:, :,x1, y0]
        Ic = world_distance_field_img[:, :,x0, y1]
        Id = world_distance_field_img[:, :,x1, y1]

        # all values above min_safe_distance are safe and should be min_safe_distance
        Ia = torch.min(Ia, self.min_safe_distance)
        Ib = torch.min(Ib, self.min_safe_distance)
        Ic = torch.min(Ic, self.min_safe_distance)
        Id = torch.min(Id, self.min_safe_distance)
        
        
        # Compute the fractional part of the points
        wa = (x1.type(torch.float32) - x) * (y1.type(torch.float32) - y)
        wb = (x1.type(torch.float32) - x) * (y - y0.type(torch.float32))
        wc = (x - x0.type(torch.float32)) * (y1.type(torch.float32) - y)
        wd = (x - x0.type(torch.float32)) * (y - y0.type(torch.float32))
        
        interpolated_values = wa * Ia + wb * Ib + wc * Ic + wd * Id
        sum_values = torch.mean(interpolated_values)
        return -sum_values

class GraphBasedConsistencyLoss(nn.Module):
    def __init__(self, variance_weight=1.0):
        super(GraphBasedConsistencyLoss, self).__init__()
        self.variance_weight = variance_weight
    
    def forward(self, paths):
        """
        Args:
            paths (torch.Tensor): Tensor of shape (Batch, Waypoints, 2) where 2 is for (x, y) coordinates.
        
        Returns:
            torch.Tensor: The graph-based consistency loss with variance penalty.
        """
        # Compute the differences between consecutive waypoints
        differences = paths[:, 1:, :] - paths[:, :-1, :]
        
        # Compute the Euclidean distances for these differences
        distances = torch.norm(differences, dim=-1)
        
        # Compute the mean distance
        mean_distance = distances.mean(dim=-1, keepdim=True)
        
        # Compute the variance of the distances
        variance = ((distances - mean_distance) ** 2).mean(dim=-1)
        
        # Sum the distances to get the consistency loss
        consistency_loss = distances.sum(dim=-1).mean()
        
        # Compute the variance penalty
        variance_penalty = variance.mean()
        
        # Total loss
        # loss = consistency_loss + self.variance_weight * variance_penalty
        loss = variance_penalty
        
        return loss




class WeightedLoss(nn.Module):
    def __init__(self, losses, weights):
        """
        Initialize the WeightedLoss module.
        
        Parameters:
        losses (list of torch.nn.Module): List of loss functions to combine.
        weights (list of float): Corresponding weights for each loss function.
        """
        super(WeightedLoss, self).__init__()
        assert len(losses) == len(weights), "Losses and weights must have the same length"
        self.losses = losses
        self.weights = weights
        self.individual_losses = [0.0 for _ in losses]
    
    def forward(self, predictions, targets):
        """
        Compute the weighted loss.
        
        Parameters:
        predictions (torch.Tensor): Model predictions.
        targets (torch.Tensor): Ground truth values.
        
        Returns:
        torch.Tensor: Computed weighted loss.
        """
        weighted_loss = 0.0
        self.individual_losses = []
        for loss_fn, weight in zip(self.losses, self.weights):
            value = loss_fn(predictions, targets)
            self.individual_losses.append(value)
            weighted_loss += weight * value
        return weighted_loss
    def get_individual_losses(self):
        """
        Get the individual losses for each loss function.
        
        Returns:
        list of float: The individual loss values.
        """
        return self.individual_losses
    def get_loss(self, indx):
        """
        Get the individual loss for a specific loss function.
        
        Parameters:
        indx (int): Index of the loss function.
        
        Returns:
        float: The individual loss value.
        """
        return self.individual_losses[indx]
# Example usage
if __name__ == "__main__":
    # Define some example loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    # Define weights for the loss functions
    losses = [mse_loss, mae_loss]
    weights = [0.7, 0.3]
    
    # Initialize the weighted loss
    weighted_loss_fn = WeightedLoss(losses, weights)
    
    # Create some example predictions and targets
    predictions = torch.tensor([0.2, 0.7, 0.1], requires_grad=True)
    targets = torch.tensor([0.0, 1.0, 0.0])
    
    # Compute the weighted loss
    loss = weighted_loss_fn(predictions, targets)
    print(f"Weighted loss: {loss.item()}")
    
    # Backward pass
    loss.backward()
    print(f"Gradients: {predictions.grad}")

    # Example paths tensor of shape (Batch, Waypoints, 2)
    paths = torch.tensor([[[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]],
                          [[1.0, 3.0], [4.0, 4.0], [5.0, 5.0]]], dtype=torch.float32)
    
    # Initialize the loss function
    loss_fn = GraphBasedConsistencyLoss()
    
    # Compute the loss
    loss = loss_fn(paths)
    
    print("Graph-Based Consistency Loss:", loss.item())
