import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
np.bool = bool # bug in wzk
import matplotlib.pyplot as plt
from wzk import sql2, trajectory
import matplotlib
import matplotlib.pyplot as plt
from .normalizer import MinMaxFeatureNormalizer

class PathsDataset(Dataset):
    def __init__(self, file, n_waypoints=20, n_dim=2, n_paths_per_world=1000, n_worlds=1, device='cuda', normalize=True, normalizer=None,n_of_waypoints=20):
        # CONSTANTS
        n_voxels = 64
        voxel_size = 10 / 64     # in m
        self.extent = [0, 10, 0, 10]  # in m
        self.MAX_X_COORDINATE = 10
        self.MAX_Y_COORDINATE = 10
        self.MIN_X_COORDINATE = 0
        self.MIN_Y_COORDINATE = 0

        # Configuration of data generation
        self.n_waypoints = n_of_waypoints  # start + 20 inner points + end
        self.n_dim = 2
        self.n_paths_per_world = 10000
        self.n_worlds = 1
        self.worlds = sql2.get_values_sql(file=file, table="worlds", values_only=False)
        self.world_images = sql2.compressed2img(img_cmp=self.worlds.img_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)


        batch_size = 32
        n_total = n_paths_per_world * n_worlds
        # batch_idx =     [0, 1, 2, 1000, 2000, 3500]
        # path_idx_for_batch = np.random.choice(np.arange(n_total), size=batch_size, replace=False)
        path_idx = np.arange(n_total)

        paths = sql2.get_values_sql(file=file, table='paths', rows=path_idx, values_only=False)
        self.worlds_indx = paths.world_i32.values

        path_coordinates = sql2.object2numeric_array(paths.q_f32.values)
        path_coordinates = path_coordinates.reshape(-1, n_waypoints, n_dim)

        self.data = torch.tensor(path_coordinates, device=device) # torch tensor
        self.og_data = self.data.clone()
        if normalize:
            # Normalize the data
            normalizer.initialize([self.MIN_X_COORDINATE, self.MIN_Y_COORDINATE], [self.MAX_X_COORDINATE, self.MAX_Y_COORDINATE], device=device)
            self.normalizer = normalizer
            self.data = self.normalizer(self.data)

        # dataset information 
        # Min value per coordinate (x, y)
        self.min_values =  torch.tensor([self.MIN_X_COORDINATE, self.MIN_Y_COORDINATE], device=device)
        # Max value per coordinate (x, y)
        self.max_values = torch.tensor([self.MAX_X_COORDINATE, self.MAX_Y_COORDINATE], device=device)
        # number of worlds in the dataset

        print(f'data shape: {self.data.shape}, min_values: {self.min_values}, max_values: {self.max_values}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            'path': self.data[idx],
            'world_indx': self.worlds_indx[idx],
            'world_img': self.world_images[self.worlds_indx[idx]]
        }
        return item

    def get_og_item(self, idx):
        item = {
            'path': self.og_data[idx],
            'world_indx': self.worlds_indx[idx],
            'world_img': self.world_images[self.worlds_indx[idx]]
        }
        return item

    def world_image_from_world_indx(self, world_indx):
        return self.world_images[world_indx]


    def get_plot_for_path(self, idx) -> matplotlib.figure.Figure:
        """
        This function plots a given path on a 2D grid with specified obstacles.

        Args:
            path (list of tuples): The path to plot, represented as a list of (x, y) coordinates.
            obstacles (list of tuples): The obstacles to plot, represented as a list of (x, y) coordinates.

        Returns:
            tuple[Figure, axes] : A tuple containing the Matplotlib figure and axis objects.
        """
        path = self.get_og_item(idx)['path'].cpu().numpy()
        world = self.world_images[idx]
        fig, ax = plt.subplots()

        ax.plot(path[:, 0], path[:, 1], 'o-')
        world_indx = self.worlds_indx[idx]

        ax.imshow(self.world_images[world_indx].T, origin='lower', extent=self.extent, cmap='binary', alpha=0.5)
        return fig, ax

    def plot_batch_of_pathes(self, batch:np.ndarray) -> matplotlib.figure.Figure:
        """
        This function plots a given path on a 2D grid with specified obstacles.

        Args:
            path (list of tuples): The path to plot, represented as a list of (x, y) coordinates.
            obstacles (list of tuples): The obstacles to plot, represented as a list of (x, y) coordinates.

        Returns:
            tuple[Figure, axes] : A tuple containing the Matplotlib figure and axis objects.
        """
        fig, ax = plt.subplots()
        for path in batch:
            ax.plot(path[:, 0], path[:, 1], 'o-')
        return fig, ax
    
    def plot_path(self, idx):
        fig, ax = self.get_plot_for_path(idx)
        plt.show()
    
if __name__=='__main__':
    file = 'D:\Desktop\ADLR\RobotPathData\dataset\RobotPathDataset\SingleSphere02_all.db'
    dataset = PathsDataset(file)
    print(f'Length of dataset: {len(dataset)}, shape of sample = {dataset[0].shape}')
    print(f'dataset MIN_X_COORDINATE: {dataset.MIN_X_COORDINATE}, dataset MAX_X_COORDINATE: {dataset.MAX_X_COORDINATE}')

    fig, axes = dataset.get_plot_for_path(500)
    # show the plot
    plt.show()
