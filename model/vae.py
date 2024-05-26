from typing import Any, List
import torch
from torch import Tensor, nn
from torch import optim
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass
    
    
class VanillaVAE(BaseVAE):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 normalizer = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.in_channels = in_channels

        self.normalizer = normalizer
        if self.normalizer:
            self.initialize_normalizer()
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.in_channels,
                                      kernel_size= 1),
                            nn.Sigmoid())

    def initialize_normalizer(self):
        self.min_values = min_values = torch.ones((64, 64)) * -3.5
        self.max_values = max_values = torch.ones((64, 64)) * 3.5

        if self.normalizer is not None:
            self.normalizer.initialize(min_values, max_values)
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        if self.normalizer is not None:
            input = self.normalizer.normalize(input)

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        if self.normalizer is not None:
            result = self.normalizer.denormalize(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

from torch import Tensor


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.save_hyperparameters()

        self.model = vae_model
        self.params = params
        self.curr_device = None


    def training_step(self, batch, batch_idx):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        train_loss = self.model.loss_function(*results,
                                              M_N = 0.0016, #al_img.shape[0]/ self.num_train_imgs,
        )

        self.log_dict({f'train/{key}': val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']
    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device

        results = self.forward(real_img)
        val_loss = self.model.loss_function(*results,
                                            M_N = 0.0016, #real_img.shape[0]/ self.num_val_imgs,
        )

        self.log_dict({f"val/val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # optimizer =  Prodigy(self.parameters(), lr=1.)
        return optimizer
    
    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)
    
    def on_train_epoch_end(self):
        NUM_SAMPLES = 8
        # Get some batch of validation data
        with torch.no_grad():
            all_preds = next(iter(self.trainer.train_dataloader))
            data = all_preds[0:NUM_SAMPLES]
            data = data.to(self.curr_device)

            # Forward pass
            results = self.forward(data)
            recon_results = results[0]
        # Visualize results
        self.visualize_results(data, recon_results, 'reconstructions/train')
        if self.current_epoch % 10 == 0:
            self.visualize_embeddings()
            
    def on_validation_epoch_end(self):
        NUM_SAMPLES = 8
        # Get some batch of validation data
        with torch.no_grad():
            all_preds = next(iter(self.trainer.val_dataloaders))
            data = all_preds[0:NUM_SAMPLES]
            data = data.to(self.curr_device)

            # Forward pass
            results = self.forward(data)
            recon_results = results[0]
        # Visualize results
        self.visualize_results(data, recon_results, tag='reconstructions/val')     
    
    def visualize_embeddings(self):
        tensorboard = self.logger.experiment
        # get 4 batches of data
        all_preds = next(iter(self.trainer.train_dataloader))
        for i in range(4):
            all_preds = torch.cat((all_preds, next(iter(self.trainer.train_dataloader))), dim=0)    
        data = all_preds.to(self.curr_device)
        # get embeddings
        mu, _ = self.model.encode(data)
        # log embeddings
        tensorboard.add_embedding(mu,
                                  global_step=self.current_epoch,
                                  tag='latent_space',
                                  label_img=data) 
        
        
    def visualize_results(self, real_img, results, tag='reconstructions'):
        tensorboard = self.logger.experiment
        n = min(real_img.size(0), 8)
        real_imgs = real_img[:n]
        recon_imgs = results[:n]
        fig, ax = plt.subplots(n,2, figsize=(10,5))
        for i in range(n):
            ax[i,0].imshow(real_imgs[i].squeeze().T.cpu().detach().numpy(), origin='lower', extent=[0, 10, 0, 10], cmap='viridis')
            ax[i,0].set_title('Original')
            ax[i,1].imshow(recon_imgs[i].squeeze().T.cpu().detach().numpy(), origin='lower', extent=[0, 10, 0, 10], cmap='viridis')
            ax[i,1].set_title('Reconstructed')
        tensorboard.add_figure(tag, fig, global_step=self.current_epoch)
        

