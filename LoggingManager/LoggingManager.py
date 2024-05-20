import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

FORMAT_DATATIME= "%Hh-%Mm-%d-%m-%Y"
class LoggingManager:
    def __init__(self, log_dir='logs', log_file='logs.log', experiement_name=None):
        if experiement_name is not None:
            log_dir = os.path.join(log_dir, experiement_name)
        self.log_dir = f'{log_dir}/{datetime.now().strftime(FORMAT_DATATIME)}'
        self.log_file = f'{log_file}'
        print(f'Logging to {self.log_dir}/{self.log_file}')
        self._setup_logging()
        self._setup_tensorboard()

    def _setup_logging(self):
        # Ensure the log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Create a logging instance
        self.logger = logging.getLogger('LoggingManager')
        self.logger.setLevel(logging.INFO)  # Set default log level

        # Create a console handler
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.ERROR)
        
        # Create a file handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, self.log_file))
        file_handler.setLevel(logging.DEBUG)
        
        # Set the log format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        # self.logger.addHandler(console_handler) # Uncomment this if u want prints
        self.logger.addHandler(file_handler)

    def _setup_tensorboard(self):
        # Create a TensorBoard log directory with a timestamp
        self.tensorboard_log_dir = os.path.join(
            self.log_dir, "tensorboard"
        )
        if not os.path.exists(self.tensorboard_log_dir):
            os.makedirs(self.tensorboard_log_dir)
        # Set up a SummaryWriter for TensorBoard
        self.summary_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)

        # Initialize a training step counter
        self.training_step = 0

    def tb_images_grid(self, images, labels=None):
        # create grid of images
        img_grid = torchvision.utils.make_grid(images)

    def plot_losses(self, train_loss, val_loss=None):
        # Plot training loss to TensorBoard
        self.summary_writer.add_scalar('Loss/train', train_loss, self.training_step)

        # Plot validation loss to TensorBoard
        if val_loss is not None:
            self.summary_writer.add_scalar('Loss/val', val_loss, self.training_step)

        # Increment the step counter after logging
        self.training_step += 1
    
    def epoch_checkpoint_images(self, fig):
          self.summary_writer.add_figure('Fig1', fig)


    def tb_diffusion_sampling(self, samples:np.ndarray, name:str):
        self.summary_writer.add_images(name, samples)

    def tb_log_figure(self, fig, name:str, step):
        self.summary_writer.add_figure(name, fig, step)
        
    def tb_log_graph(self, model, input_tensor=None):
        self.summary_writer.add_graph(model, input_tensor)
        
    def tb_log_embedding_space(self, features, metadata, label_img ,tag='embedding_space'):
        self.summary_writer.add_embedding(features, metadata, label_img, tag)
        

    def log_message(self, message, print_message=False,level='info'):
        # Log a message at the specified level
        if level.lower() == 'info':
            self.logger.info(message)
        elif level.lower() == 'debug':
            self.logger.debug(message)
        elif level.lower() == 'warning':
            self.logger.warning(message)
        elif level.lower() == 'error':
            self.logger.error(message)
        else:
            self.logger.info(message)
        
        if print_message:
            print(message)


    def get_tensorboard_log_dir(self):
        return self.tensorboard_log_dir

    def log_config(self, config:dict):
        """logs the configuration dictonary to the log file

        Args:
            config (dict): key value pairs of configuration
        """
        self.logger.info('Configuration:')
        for key, value in config.items():
            self.logger.info(f'{key}: {value}')

    
    # CHECKPOINT MODEL
    def save_checkpoint(self, model, optimizer, scheduler,epoch, loss, filename='checkpoint.pth'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'scheduler_state_dict': scheduler.state_dict(),
            'training_step': self.training_step,
        }
        full_filename = f'model-epoch_{epoch}-loss_{loss:.3f}-time_{datetime.now().strftime(FORMAT_DATATIME)}' # Add loss and epoch to filename
        torch.save(checkpoint, os.path.join(self.log_dir, full_filename))
        self.logger.info(f'Checkpoint saved to {os.path.join(self.log_dir, full_filename)}')
    
    def load_checkpoint(self, model, optimizer, scheduler,filename='checkpoint.pth'):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'training_step' in checkpoint:
            self.training_step = checkpoint['training_step']
        self.logger.info(f'Checkpoint loaded from {filename}')
        return model, optimizer, scheduler,epoch, loss



if __name__=='__main__':
    print('LoggingManager class')
    logging_manager = LoggingManager()  
    logging_manager.log_message('info', 'LoggingManager class')
    logging_manager.plot_losses(0.1, 0.2)
