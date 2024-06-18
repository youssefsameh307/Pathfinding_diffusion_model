import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
import numpy as np
class Diffusion():

    def __init__(self, input_shape=(20, 2),noise_steps=100, beta_start=1e-4, beta_end=3e-4, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.input_shape = input_shape
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.device = device

    def prepare_noise_schedule(self, scheduler_type='liner'):
        if scheduler_type=='liner':
            return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        if scheduler_type=='cosine':
            return self.betas_for_alpha_bar(self.noise_steps, max_beta=0.999, alpha_transform_type="cosine")
            
    
    def betas_for_alpha_bar(
        self,
        num_diffusion_timesteps,
        max_beta=0.999,
        alpha_transform_type="cosine",
    ):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
        (1-beta) over time from t = [0,1].

        Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
        to that part of the diffusion process.


        Args:
            num_diffusion_timesteps (`int`): the number of betas to produce.
            max_beta (`float`): the maximum beta to use; use values lower than 1 to
                        prevent singularities.
            alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                        Choose from `cosine` or `exp`

        Returns:
            betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
        """
        if alpha_transform_type == "cosine":

            def alpha_bar_fn(t):
                return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

        elif alpha_transform_type == "exp":

            def alpha_bar_fn(t):
                return np.exp(t * -12.0)

        else:
            raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)


    def noise_input(self, x, t):
        sqrt_alpha = torch.sqrt(self.alpha[t])[:, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus_alpha * Ɛ, Ɛ 
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    @torch.no_grad()
    def sample(self, model,n, cfg_scale=0,save_rate=20, cond=None,mu=0, sigma=1):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        x = mu + sigma *  torch.randn((n, *self.input_shape)).to(self.device)
        if cond is not None:
            cond = cond.to(self.device)

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = model(x, t, cond)
            if cfg_scale:
                uncondditional_predicted_noise = model(x, t, cond=None)
                predicted_noise = torch.lerp(uncondditional_predicted_noise, predicted_noise, cfg_scale)

            alpha = self.alpha[t][:, None, None]

            alpha_hat = self.alpha_hat[t][:, None, None]

            beta = self.beta[t][:, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x)
        intermediate = torch.stack(intermediate)
        x = x
        return x, intermediate


    @torch.no_grad()
    def sample_with_constraints(self, model, n, constraints={}, cond=None, cfg_scale=0,normalizer=None,save_rate=20, mu=0, sigma=1):
        """sample with constraints. constraints is a dictionary with keys as timesteps and values as the constraints at that timestep. 
        Sampling will reset the constraints at the given timesteps to the given values at each timestep. 

        mu and sigma are choose like this as the word is from 0 to 10. and it is uniformaly distributed. 
        Args:
            model (torch.nn): torch model
            n (int): number of samples to generate
            constraints (dict, optional): {timestamp: value}. Defaults to {}.
            save_rate (int, optional): _description_. Defaults to 20.

        Returns:
            tuple: samples, intermediate diffusions
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        if cond is not None:
            cond = cond.to(self.device)
        # normalize the constraints
        if normalizer is not None:
            for t, value in constraints.items():
                constraints[t] = normalizer(value)
        
        x = mu + sigma * torch.randn((n, *self.input_shape)).to(self.device)
        # set constraints
        for t, value in constraints.items():
            
            x[:, t, :] = value

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = model(x, t, cond)
            if cfg_scale > 0:
                uncondditional_predicted_noise = model(x, t, cond=None)
                predicted_noise = torch.lerp(uncondditional_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None, None]

            alpha_hat = self.alpha_hat[t][:, None, None]

            beta = self.beta[t][:, None, None]

            if i > 1:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            for t, value in constraints.items():
                x[:, t, :] = value
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x.detach().cpu().numpy())
        intermediate = np.stack(intermediate)
        model.train()
        x = x.cpu().detach().numpy()
        return x, intermediate

    @torch.no_grad()
    def sample_from_paths(self, model, n, paths, cond=None, cfg_scale=0,normalizer=None,save_rate=20, mu=0, sigma=0.1):
        """sample with constraints. constraints is a dictionary with keys as timesteps and values as the constraints at that timestep.
        
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        if cond is not None:
            cond = cond.to(self.device)

        # sample from the paths by setting start and end points as start and end of the path
        

        x = mu + sigma * torch.randn((n, *self.input_shape)).to(self.device)
        # set constraints
        start_pos = x[:, 0, :] = paths[:, 0]
        end_pos = x[:, -1, :] = paths[:, -1]

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = model(x, t, cond, start_pos, end_pos)
            if cfg_scale > 0:
                uncondditional_predicted_noise = model(x, t, cond=None)
                predicted_noise = torch.lerp(uncondditional_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None, None]

            alpha_hat = self.alpha_hat[t][:, None, None]

            beta = self.beta[t][:, None, None]

            if i > 1:
                noise = torch.randn_like(x)
                # noise = torch.zeros_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha))  * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # set constraints
            x[:, 0, :] = paths[:, 0]
            x[:, -1, :] = paths[:, -1]
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x.detach().cpu().numpy())
        intermediate = np.stack(intermediate)
        model.train()
        x = x.cpu().detach().numpy()
        return x, intermediate
    @torch.no_grad()
    def sample_from_paths_with_constraints(self, model, n, paths, constraints,cond=None, cfg_scale=0,normalizer=None,save_rate=20, mu=0, sigma=0.1):
        """sample with constraints. constraints is a dictionary with keys as timesteps and values as the constraints at that timestep.
        
        """
        logging.info(f"Sampling {n} new images....")
        model.eval()
        if cond is not None:
            cond = cond.to(self.device)
            
        if normalizer is not None:
            for t, value in constraints.items():
                constraints[t] = normalizer(value)

        # sample from the paths by setting start and end points as start and end of the path
        

        x = mu + sigma * torch.randn((n, *self.input_shape)).to(self.device)
        # set constraints
        start_pos = x[:, 0, :] = paths[:, 0]
        end_pos = x[:, -1, :] = paths[:, -1]
        for t, value in constraints.items():
            x[:, t, :] = value

        # array to keep track of generated steps for plotting
        intermediate = [] 
        for i in tqdm(reversed(range(1, self.noise_steps)), position=0):

            t = (torch.ones(n) * i).long().to(self.device)

            predicted_noise = model(x, t, cond, start_pos, end_pos)
            if cfg_scale > 0:
                uncondditional_predicted_noise = model(x, t, cond=None)
                predicted_noise = torch.lerp(uncondditional_predicted_noise, predicted_noise, cfg_scale)
            alpha = self.alpha[t][:, None, None]

            alpha_hat = self.alpha_hat[t][:, None, None]

            beta = self.beta[t][:, None, None]

            if i > 1:
                noise = torch.randn_like(x)
                # noise = torch.zeros_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha))  * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # set constraints
            x[:, 0, :] = paths[:, 0]
            x[:, -1, :] = paths[:, -1]
            for t, value in constraints.items():
                x[:, t, :] = value
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x.detach().cpu().numpy())
        intermediate = np.stack(intermediate)
        model.train()
        x = x.cpu().detach().numpy()
        return x, intermediate



if __name__ == '__main__':
    pass




