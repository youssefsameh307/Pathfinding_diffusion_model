import torch
import matplotlib.pyplot as plt
from lightning.pytorch.callbacks import Callback
from .diffusion import plot_diffusions, plot_noise_and_predicted_noise

class VisualizeProgress(Callback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps
    def on_train_start(self, trainer, pl_module):
        print("Training is starting")
        for i in range(5):
            sample_batch = next(iter(trainer.train_dataloader))
            points = 10
            # create plots for the entire diffusion process to make sure we get a feeling for betas
            diffusion = pl_module.diffusion
            timesteps = torch.linspace(1, pl_module.config['noise_steps']-1,steps=points, dtype=torch.int).to(pl_module.device)
            first_element = sample_batch[pl_module.path_type][0].unsqueeze(0)  # Adds a batch dimension
            replicated_batch = first_element.repeat(points, 1, 1)
            # Assign it back to the batch
            sample_batch['offset_path_replicated'] = replicated_batch
            sample_batch['offset_path_replicated'].shape
            x_t, noise = diffusion.noise_input(sample_batch['offset_path_replicated'],timesteps)
            # PLOT INPUT VS NOISED INPUT
            # create a subpolt for 2 graphs
            fig, axs = plt.subplots(points, 2, figsize=(16, 16))
            for i in range(points):
                # the 2nd will be the path + some noise and its distance field + loss value
                path_noised = sample_batch['offset_path_replicated'][i].cpu().detach().numpy()
                axs[i, 0].plot(path_noised[:, 0], path_noised[:, 1], '-o', color='red')
                # the 3rd will be the path + some noise and its distance field + loss value
                path_noised = x_t[i].cpu().detach().numpy()
                axs[i, 1].plot(path_noised[:, 0], path_noised[:, 1], '-o', color='red', label=f'timestep {timesteps[i].item()}')
            
            trainer.logger.experiment.add_figure('diffusion betas', fig, i);

    def on_train_end(self, trainer, pl_module):
        print("Training is ending")

    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step) % self.log_every_n_steps == 0:
            try:
                logging_cache = pl_module.logging_cache
                x, noise, pred_noise, timesteps_choosen = logging_cache['x_path'], logging_cache['noise'], logging_cache['pred_noise'], logging_cache['timesteps_choosen']
                # show predicted noise vs real noise
                plot = plot_noise_and_predicted_noise(x, noise, pred_noise, timesteps_choosen, normalizer=pl_module.config['normalizer'])
                trainer.logger.experiment.add_figure('generated_images', plot, trainer.global_step)
            except Exception as e:
                # print(f'Error in on_train_batch_end {e}')
                pass

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (trainer.global_step) % self.log_every_n_steps == 0:
            try:
                logging_cache = pl_module.logging_cache
                samples, intermediates, x_path, x_world_img = logging_cache['samples'], logging_cache['intermediates'], logging_cache['x_path'], logging_cache['x_world_img']
                # plot diffusions 
                plot = plot_diffusions(intermediates, world_imgs=x_world_img, normalizer=pl_module.config['normalizer'], true_paths=x_path);
                trainer.logger.experiment.add_figure('no-inpainting', plot, trainer.global_step)

                # plot diffusions on same path to see model diversity
                samples_for_same_point, intermediates_for_same_point,x_path_for_same_point, x_world_img_for_same_point = logging_cache['samples_for_same_point'], logging_cache['intermediates_for_same_point'], logging_cache['x_path_for_same_point'], logging_cache['x_world_img_for_same_point']
                plot = plot_diffusions(intermediates_for_same_point, world_imgs=x_world_img_for_same_point, normalizer=pl_module.config['normalizer'], true_paths=x_path_for_same_point);
                trainer.logger.experiment.add_figure('diversity-diffusion', plot, trainer.global_step)

                # plot diffusions using inpaitning 
                samples_inpainting, intermediates_inpainting, x_path_inpainting, x_world_img_inpainting = logging_cache['samples_inpainting'], logging_cache['intermediates_inpainting'], logging_cache['x_path_inpainting'], logging_cache['x_world_img_inpainting']
                constraints = logging_cache['constraints']
                plot = plot_diffusions(intermediates_inpainting, world_imgs=x_world_img_inpainting, normalizer=pl_module.config['normalizer'], true_paths=x_path_inpainting, constraints=constraints);
                trainer.logger.experiment.add_figure('inpainting', plot, trainer.global_step)
            except Exception as e:
                print(f'Error in on_validation_batch_end {e}')

