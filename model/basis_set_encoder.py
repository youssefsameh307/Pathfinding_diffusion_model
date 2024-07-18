import torch
import torch.nn as nn


def generate_grid_basis(grid_size=32, n_dims=3, minv=-1.0, maxv=1.0, device='cpu'):
    """ Generate d-dimensional grid BPS basis using PyTorch

    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid
    device: str
        PyTorch device to use ('cpu' or 'cuda')

    Returns
    -------
    basis: torch tensor [grid_size**n_dims, n_dims]
        n-d grid points
    """
    linspaces = [torch.linspace(minv, maxv, grid_size, device=device) for _ in range(n_dims)]
    coords = torch.meshgrid(*linspaces)
    basis = torch.cat([coord.reshape(-1, 1) for coord in coords], dim=1)

    return basis

class BPS_Encoder(nn.Module):
    def __init__(self, n_bps_points, n_dims, minv, maxv, device):
        super(BPS_Encoder, self).__init__()
        grid_size = int(round(pow(n_bps_points, 1 / n_dims)))
        self.basis_set = generate_grid_basis(grid_size, n_dims, minv, maxv, device)

    def forward(self, batch_world_binary_img): 
        # batch_world_binary_img: [batch_size, height, width] or [batch_size, 1, height, width]
        if len(batch_world_binary_img.shape) == 4:
            batch_world_binary_img = batch_world_binary_img.squeeze(1)
        with torch.no_grad():
            batch_size = batch_world_binary_img.shape[0]
            values = []
            
            for i in range(batch_size):
                img_indices = torch.argwhere(batch_world_binary_img[i])
                dist = torch.norm(img_indices.unsqueeze(1).float() - self.basis_set.unsqueeze(0).float(), dim=2)
                knn = dist.topk(1, largest=False, dim=0)
                values.append(knn.values[0, :])
            
            values = torch.stack(values, dim=0)
        return values

if __name__ == "__main__":
    # Example dataset with binary images
    dataset = {
        6006: {'world_img': [[0, 1, 0], [0, 1, 1], [0, 0, 0]]},
        1007: {'world_img': [[1, 0, 0], [0, 0, 0], [1, 1, 0]]}
    }
    
    # Create binary images tensors
    world_binary_img_1 = torch.tensor(dataset[6006]['world_img'], device='cuda')
    world_binary_img_2 = torch.tensor(dataset[1007]['world_img'], device='cuda')
    batch_world_binary_img = torch.stack([world_binary_img_1, world_binary_img_2], dim=0)
    
    # Define parameters for the basis set
    n_bps_points = 64
    n_dims = 2
    minv = 0
    maxv = 64
    device = 'cuda'
    
    # Initialize and run the encoder
    encoder = BPS_Encoder(n_bps_points=n_bps_points, n_dims=n_dims, minv=minv, maxv=maxv, device=device)
    bps_encoding = encoder(batch_world_binary_img)
    
    print(bps_encoding.shape)
    print(bps_encoding)
