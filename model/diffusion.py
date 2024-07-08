import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
import logging
import numpy as np
import sched
from google.cloud.aiplatform.constants import schedule
class Diffusion():

    def __init__(self, input_shape=(20, 2),noise_steps=100, beta_start=1e-4, beta_end=0.02, device="cuda", scheduler_type=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.input_shape = input_shape
        self.device = device
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.device = device

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_input(self, x, t):
        sqrt_alpha = torch.sqrt(self.alpha[t])[:, None, None]
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alpha[t])[:, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha * x + sqrt_one_minus_alpha * Ɛ, Ɛ
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    @torch.no_grad()
    def sample(self, model,n, cfg_scale=3,save_rate=20, cond=None,mu=0, sigma=1):
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
    def sample_with_constraints(self, model, n, constraints={}, cond=None, cfg_scale=3,normalizer=None,save_rate=20, mu=0, sigma=1):
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
            for t, value in constraints.items():
                x[:, t, :] = value
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x)
        intermediate = torch.stack(intermediate)
        x = x
        return x, intermediate

    @torch.no_grad()
    def sample_from_paths(self, model, n, paths, cond=None, cfg_scale=0,normalizer=None,save_rate=20, mu=0, sigma=1):
        """sample with constraints. constraints is a dictionary with keys as timesteps and values as the constraints at that timestep. 
        Sampling will reset the constraints at the given timesteps to the given values at each timestep. 

        mu and sigma are choose like this as the word is from 0 to 10. and it is uniformaly distributed. 
        Args:
            model (torch.nn): torch model
            paths torch.tensor: shape (n, waypoints, 2)
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
            else:
                noise = torch.zeros_like(x)

            x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
            # set constraints
            x[:, 0, :] = paths[:, 0]
            x[:, -1, :] = paths[:, -1]
            # save intermediate images
            if i % save_rate ==0 or i==self.noise_steps or i<8:
                intermediate.append(x)
        intermediate = torch.stack(intermediate)
        x = x
        return x, intermediate

if __name__ == '__main__':
    pass




