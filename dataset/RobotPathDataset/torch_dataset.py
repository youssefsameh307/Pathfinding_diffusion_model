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
from .obstacle_distance import img2dist_img, img2grad


class PathsDataset(Dataset):
    def __init__(self, file, n_waypoints=20, n_dim=2, n_paths_per_world=1000, n_worlds=1, device='cuda', normalizer=None, single_world_dataset=False):
        # CONSTANTS
        self.voxel = n_voxels = 64
        self.voxel_size = 10 / 64     # in m
        self.extent = [0, 10, 0, 10]  # in m
        self.MAX_X_COORDINATE = 10
        self.MAX_Y_COORDINATE = 10
        self.MIN_X_COORDINATE = 0
        self.MIN_Y_COORDINATE = 0

        # Configuration of data generation
        self.n_waypoints = n_waypoints  # start + 20 inner points + end
        self.n_dim = n_dim
        self.n_paths_per_world = n_paths_per_world
        self.n_worlds = n_worlds
        self.all_worlds = sql2.get_values_sql(file=file, table="worlds", values_only=False)
        self.all_world_images = sql2.compressed2img(img_cmp=self.all_worlds.img_cmp.values, shape=(n_voxels, n_voxels), dtype=bool)

        # always 1000 paths belong to one world
        # 0...999     -> world 0
        # 1000...1999 -> world 1
        # 2000...2999 -> world 2
        self.MAX_PATHS_PER_WORLD = 1000
        self.MAX_WORLDS = 5000
        # Create indexes for the paths
        world_idx, path_idx = self.get_indecies(n_paths_per_world, n_worlds)
        # path_idx = np.random.choice(np.arange(self.MAX_WORLDS*n_worlds), size=n_worlds*n_paths_per_world, replace=False)
        if single_world_dataset:
            path_idx = np.arange(n_paths_per_world)
        paths = sql2.get_values_sql(file=file, table='paths', rows=path_idx, values_only=False)
        self.worlds_indx = paths.world_i32.values
        
        # get distance field images
        self.world_distance_field_images = self.preprocess_world_images(self.all_world_images, self.worlds_indx,self.voxel_size) # (n_worlds, 64, 64)
        # add channel dimension
        self.world_distance_field_images = {k: v.unsqueeze(0) for k, v in self.world_distance_field_images.items()}
        # only keep the world images that are needed
        self.world_images = self.all_world_images[self.worlds_indx]

        path_coordinates_og = sql2.object2numeric_array(paths.q_f32.values)
        path_coordinates_og = path_coordinates_og.reshape(-1, 20, n_dim) # reshape to (n_paths, n_waypoints, n_dim) = (n_total, 20, 2) because 20 is the default number of waypoints
        # map this path to n_waypoints
        
        path_coordinates = self.preprocess_path_coordinates(path_coordinates_og) # (n_total, n_of_waypoints, 2)
        
        self.data = torch.tensor(path_coordinates, device=device) # torch tensor
        self.og_data = self.data.clone()
        if normalizer:
            # Normalize the data
            normalizer.initialize([self.MIN_X_COORDINATE, self.MIN_Y_COORDINATE], [self.MAX_X_COORDINATE, self.MAX_Y_COORDINATE], device=device)
            self.normalizer = normalizer
            self.data = self.normalizer(self.data)

        # Create relative_path_data
        self.relative_path_data, self.straight_line_path_data = self.relative_paths(self.data)
        self.relative_path_data, self.straight_line_path_data = torch.tensor(self.relative_path_data, device=device, dtype=torch.float32), torch.tensor(self.straight_line_path_data, device=device, dtype=torch.float32)
        

        # dataset information 
        # Min value per coordinate (x, y)
        self.min_values =  torch.tensor([self.MIN_X_COORDINATE, self.MIN_Y_COORDINATE], device=device)
        # Max value per coordinate (x, y)
        self.max_values = torch.tensor([self.MAX_X_COORDINATE, self.MAX_Y_COORDINATE], device=device)
        # number of worlds in the dataset

        print(f'data shape: {self.data.shape}, min_values: {self.min_values}, max_values: {self.max_values}, n_worlds:{len(np.unique(self.worlds_indx))}, samples: {len(self.data)}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {
            'path': self.data[idx],
            'og_path': self.og_data[idx],
            'world_indx': self.worlds_indx[idx],
            'world_img': self.world_images[idx],
            'world_distance_field_img': self.world_distance_field_images[self.worlds_indx[idx]],
            'relative_path': self.relative_path_data[idx],
            'straight_line_path': self.straight_line_path_data[idx],
        }
        return item

    def get_og_item(self, idx):
        item = {
            'path': self.og_data[idx],
            'world_indx': self.worlds_indx[idx],
            'world_img': self.world_images[idx],
            'world_distance_field_img': self.world_distance_field_images[self.worlds_indx[idx]],
            'relative_path': self.relative_path_data[idx],
            'straight_line_path': self.straight_line_path_data[idx],
        }
        return item
    
    def get_distance_field_image(self, idx, voxel_size=10/64):
        return img2dist_img(img=self.world_images[idx], voxel_size=voxel_size, add_boundary=False)
    
    ### Preprocessing functions ###
    def relative_paths(self, paths:torch.Tensor):
        """
        This function computes the relative paths between the straignt line with n_waypoints and the offset of the path given.

        Args:
            paths (Tensor): The paths to compute relative paths for, with shape (batch, timesteps, features).

        Returns:
            Tensor: The relative paths, with shape (batch, timesteps, features).
        """
        paths = paths.detach().cpu().numpy()
        # get the start and end point of the path
        path_with_only_start_and_end = paths[:, [0, -1], :]
        # get the straight line between the start and end point
        straight_line = trajectory.get_path_adjusted(path_with_only_start_and_end, n=self.n_waypoints)
        offset_paths = paths - straight_line
        return offset_paths, straight_line


    def preprocess_world_images(self, world_images, world_indx,voxel_size):
        return self.transform_world_images_to_distance_field(world_images,world_indx, voxel_size)
    
    def transform_world_images_to_distance_field(self, world_images, world_indx, voxel_size):
        """Transforms world images to distance field images. Does not transform all of them, only the ones that are needed for preformance needed


        Args:
            world_images (numpy.ndarray): Array of world images.
            world_indx (numpy.ndarray): Array of world indices.
            voxel_size (float): Voxel size for distance field conversion.

        Returns:
            numpy.ndarray: Array of distance field images.
        """
        # loop over each unique world and create a distance field image
        world_distance_field_images = {}
        for indx in np.unique(world_indx):
            world_distance_field_images[indx] = torch.tensor(img2dist_img(img=world_images[indx], voxel_size=voxel_size, add_boundary=True), dtype=torch.float32)
            print(f'world_indx: {indx}, world_distance_field_images: {world_distance_field_images[indx].shape}')
        return world_distance_field_images
        
        
    def preprocess_path_coordinates(self, path_coordinates):
            # do it but split up in chunks
            if len(path_coordinates) < 100:
                num_of_chunks = 1
            else:
                num_of_chunks = 100
            chunk_size = len(path_coordinates) // num_of_chunks
            path_coordinates = np.array_split(path_coordinates, num_of_chunks)
            for i in range(num_of_chunks):
                path_coordinates[i] = trajectory.get_path_adjusted(path_coordinates[i], n=self.n_waypoints)
            all_paths =  np.concatenate(path_coordinates, axis=0)
            return torch.tensor(all_paths.astype(np.float32))

    def world_image_from_world_indx(self, world_indx):
        return self.all_world_images[world_indx]


    def get_indecies(self, n_paths_per_world, n_worlds):
        # choose n random worlds from all worlds
        world_idx = np.random.choice(np.arange(len(self.all_worlds)), size=n_worlds, replace=False)
        # create a list of paths for each world
        path_idx = []
        for i in world_idx:
            start_of_range = i*n_paths_per_world
            end_of_range = (i+1)*n_paths_per_world
            range_of_paths = np.arange(start_of_range, end_of_range)
            world_path_idx = np.random.choice(range_of_paths, size=n_paths_per_world, replace=False)
            path_idx.extend(world_path_idx)
        return world_idx, path_idx
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
        world = self.all_world_images[idx]
        fig, ax = plt.subplots()

        ax.plot(path[:, 0], path[:, 1], 'o-')
        world_indx = self.worlds_indx[idx]

        ax.imshow(self.all_world_images[world_indx].T, origin='lower', extent=self.extent, cmap='binary', alpha=0.5)
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
