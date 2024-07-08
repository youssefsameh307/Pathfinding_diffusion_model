
import numpy as np
import torch
from torch import nn
from typing import Any
from model.diffusion import Diffusion
import lightning.pytorch as L
from model.encoder import VAEEncoder
from model.models import TemporalUnet
from model.vae import VAEXperiment
from utils.path import encoded_path_to_real_path
from model import ObstacleFreePathLoss, GraphBasedConsistencyLoss
from utils.visualize.diffusion import plot_diffusions
from model.transfomer_arch import TransformerEncoder
from model.model_blocks import Sine
from utils.multi_loss import ema_losses


class PathDiffusionModel(L.LightningModule):

    def __init__(self, config_dict, sample_input_batch, encoder=None, path_type='offset_path', cond_type='world_distance_field_img', log_every_n_steps=1000):
        super().__init__()
        
        self.config = config_dict
        
        sample_input = sample_input_batch[path_type]
        self.transition_dim = sample_input.shape[-1]
        self.representation_dim = config_dict['representation_dim']
        self.path_type = path_type
        self.cond_type = cond_type
        self.n_waypoints = config_dict['n_waypoints']
        
        self.world_encoder = encoder

        # Encoder
        # world encoder
        # Import pretrained vae model
        full_vae = VAEXperiment.load_from_checkpoint(self.config['world_encoder']['ckpt_path'])
        if self.config['world_encoder']['end2end_train'] == False:
            full_vae.freeze()
        vae_world_encoder = full_vae.model
        self.world_encoder = VAEEncoder(vae_world_encoder)
        # start_end position encoder input: (batch, transition_dim, n_waypoints) output: (batch, representation_dim, n_waypoints)
        self.start_end_encoder = nn.Sequential(
            nn.Linear(self.transition_dim, self.representation_dim),
            Sine() if self.config['transfomer']['use_sin_activation'] else nn.ReLU(),
        )
        
        # Loss functions
        self.obstacle_avoidance_loss_fn = ObstacleFreePathLoss(normalizer=self.config['normalizer'])
        self.graph_based_consistency_loss_fn = GraphBasedConsistencyLoss()
        
        if self.config['model_type'] == 'transfomer':
            # Create model
            self.model = TransformerEncoder(input_dim=self.transition_dim, output_dim=self.transition_dim,
                                            cond_dim=self.config['representation_dim'],
                                            d_model=self.representation_dim,
                                            nhead=self.config['transfomer']['num_heads'],
                                            num_layers=self.config['transfomer']['num_layers'],
                                            use_sin_activation=self.config['transfomer']['use_sin_activation'],
                                            dim_feed_forward_ratio=self.config['transfomer']['dim_feed_forward_ratio'],
                                            use_adaln=self.config['transfomer']['use_adaln'])
        elif self.config['model_type'] == 'u_net':
            # create model2 TemporalU-net
            self.model = TemporalUnet(horizon=self.config['noise_steps'],
                                    transition_dim=self.transition_dim,
                                    dim=self.config['representation_dim'],
                                    dim_mults=self.config['u_net']['dim_mults'],
                                    encoder=self.world_encoder,
                                    attention=self.config['u_net']['attention'],
                                    device=self.device)
        else:
            raise NotImplementedError(f"Model type {self.config['model_type']} not implemented")
        # diffusion process
        self.diffusion = Diffusion(input_shape=sample_input.shape, noise_steps=self.config['noise_steps'], beta_end=config_dict['beta_end'], beta_start=config_dict['beta_start'], scheduler_type=config_dict['diffusion_scheduler'])




        self.save_hyperparameters()

        ### Logging ###
        self.log_every_n_steps = log_every_n_steps

    def training_step(self, batch, batch_idx):
        # extract data
        # x_path: The path data extracted from the batch, typically a sequence of coordinates or actions.
        # x_world_indx: An index or identifier for the specific world or environment from which the path data is extracted.
        # x_world_img: An image representation of the world or environment associated with the path data in binary.
        # x_world_img_distance_field: A signed distance field image derived from x_world_img, representing distances within the environment.
        # x_offset_path: offset of each waypoint from the straight line path from start to end
        # x_straighline_path: A simplified or idealized version of the path, typically represented as a straight line from start to end point.
        x_path, x_world_indx, x_world_img, x_world_img_distance_field, x_offset_path, x_straigh_path = batch['path'], batch['world_indx'], batch['world_img'], batch['world_distance_field_img'], batch['offset_path'], batch['straight_line_path']
        # x: The selected path data for processing, determined by the 'path_type' specified in the batch. This could be any of the above path representations depending on the context.        
        x = batch[self.path_type]
        world_cond = batch[self.cond_type]
        start_pos = x[:, 0,:]
        end_pos = x[:, -1,:] 
        start_and_end_pos = x[:, [0, -1],:]
        # encode conditionings
        # encoder world 
        world_encoding = self.world_encoder(world_cond)
        # start position encoding
        start_end_encoding = self.start_end_encoder(start_and_end_pos)
        # fuse the conditionings world encoding + start encoding + end encoding
        # cond = world_encoding + start_end_encoding[:, 0,:] + start_end_encoding[:, 1,:]
        cond = torch.zeros_like(world_encoding)
        # sample timesteps
        timesteps_choosen = self.diffusion.sample_timesteps(x.shape[0])
        # set all timesteps to 1
        x_t, noise = self.diffusion.noise_input(x, timesteps_choosen) 
        # get model to predict the noise
        if self.config['predict_noise']:
            if self.config['model_type'] == 'transfomer':
                pred_noise = self.model(x_t, timesteps_choosen, cond)
            elif self.config['model_type'] == 'u_net':
                pred_noise = self.model(x_t, timesteps_choosen, world_cond, start_pos, end_pos)
            x_pred = x + pred_noise  # TODO this has to be wrong. anyway
            loss_reconstruction = nn.SmoothL1Loss()(noise, pred_noise)
        else:
            x_pred = self.model(x_t, timesteps_choosen, cond)
            pred_noise = x_t - x_pred
            loss_reconstruction = nn.SmoothL1Loss()(x, x_pred)

        # get predicted path depending on the noise
        pred_path = encoded_path_to_real_path(encoded_path=x_pred, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos)
        loss_obstacle_avoidance = self.obstacle_avoidance_loss_fn(pred_path, x_world_img_distance_field)
        loss_graph_based_consistency = self.graph_based_consistency_loss_fn(pred_path)
        loss_1 = loss_reconstruction
        # loss_2 = 1 + self.config['loss_weights']['MSE']*loss_reconstruction + self.config['loss_weights']['ObstacleFreePathLoss']*loss_obstacle_avoidance +  self.config['loss_weights']['GraphBasedConsistencyLoss'] * loss_graph_based_consistency
        # loss = ema_losses(loss_1, loss_2, self.global_step, self.config['multi_loss_warm_up_steps'])
        loss = loss_1
        
        # log 
        model_cache = self.model.cache
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_loss/noise_prediction', loss_reconstruction)
        self.log('train_loss/obstacle_avoidance', loss_obstacle_avoidance)
        self.log('train_loss/graph_based_consistency', loss_graph_based_consistency)
        # # plot the noise and predicted noise histograms
        if (self.global_step) % self.log_every_n_steps == 0:
            self.logger.experiment.add_histogram('histograms/noise', noise, self.global_step)
            self.logger.experiment.add_histogram('histograms/pred_noise', pred_noise, self.global_step)
            self.logger.experiment.add_histogram('histograms/noise-pred_noise', noise - pred_noise, self.global_step)
            self.logger.experiment.add_histogram('histograms/x_t', x_t, self.global_step)
            self.logger.experiment.add_histogram('histograms/x_pred', x_pred, self.global_step)
            self.logger.experiment.add_histogram('histograms/x', x, self.global_step)
            self.logger.experiment.add_histogram('histograms/world_encoding', world_encoding, self.global_step)
            self.logger.experiment.add_histogram('histograms/start_end_encoding', start_end_encoding, self.global_step)
            self.logger.experiment.add_histogram('histograms/cond', cond, self.global_step)
            # internal model caching
            if 'emb_timestep' in model_cache:
                self.logger.experiment.add_histogram('histograms/model_emb_timestep', model_cache['emb_timestep'], self.global_step)
                self.logger.experiment.add_histogram('histograms/model_emb_cond_after_dense', model_cache['emb_cond_after_dense'], self.global_step)
                self.logger.experiment.add_histogram('histograms/model_emb_normalized', model_cache['emb_normalized'], self.global_step)
            # self.logger.experiment.add_histogram('histograms/adaLN-1 weights', model_cache['block_1-adaLN'], self.global_step)
            # self.logger.experiment.add_histogram('histograms/adaLN-3 weights', model_cache['block_3-adaLN'], self.global_step)
            self.log('optim/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
            if 'd' in self.trainer.optimizers[0].param_groups[0]:
                self.log('optim/d', self.trainer.optimizers[0].param_groups[0]['d'])

        self.logging_cache = {
            'pred_path': pred_path,
            'x_path': x_path,
            'x_world_img': x_world_img,
            'x_world_img_distance_field': x_world_img_distance_field,
            'x_offset_path': x_offset_path,
            'timesteps_choosen': timesteps_choosen,
            'noise': noise,
            'pred_noise': pred_noise,
        }
        return loss

    def configure_optimizers(self):
        optimizer = self.config['optimizer'](self.parameters(), **self.config['optimizer_kwargs'])
        scheduler = self.config['scheduler'](optimizer, **self.config['scheduler_kwargs'])
        return {
            'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'train_loss',
       }
    
    def validation_step(self, batch, batch_idx):
        ################################################
        ### GENERATE RANDOM PATHS WITHOUT INPAINTING ###
        ################################################
        
        x_path, x_world_indx, x_world_img, x_world_img_distance_field, x_offset_path, x_straigh_path = batch['path'], batch['world_indx'], batch['world_img'], batch['world_distance_field_img'], batch['offset_path'], batch['straight_line_path']
        x = batch[self.path_type]
        world_cond = batch[self.cond_type]
        start_pos = x[:, 0,:]
        end_pos = x[:, -1,:]
        start_and_end_pos = x[:, [0, -1],:]
        # encode conditionings
        # encoder world 
        world_encoding = self.world_encoder(world_cond)
        # start position encoding
        start_end_encoding = self.start_end_encoder(start_and_end_pos)
        # fuse the conditionings world encoding + start encoding + end encoding
        if self.config['model_type'] == 'transfomer':
            cond = world_encoding + start_end_encoding[:, 0,:] + start_end_encoding[:, 1,:]
        elif self.config['model_type'] == 'u_net':
            cond = world_cond
        # diffuse entire path 
        samples_offset , intermediates_offset = self.diffusion.sample(self.model, n=x_path.shape[0], cond=cond)
        # samples, intermediates = self.diffusion.sample_from_paths(model, n=n, paths=paths, cond=cond, normalizer=CONFIG['normalizer'])
        intermediates = torch.stack([encoded_path_to_real_path(encoded_path=intermediate_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos) for intermediate_offset in intermediates_offset])        
        samples = encoded_path_to_real_path(encoded_path=samples_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos)
        # loss to see the quality of the paths
        loss_obstacle_avoidance = self.obstacle_avoidance_loss_fn(samples, x_world_img_distance_field)
        loss_graph_based_consistency = self.graph_based_consistency_loss_fn(samples)
        loss = self.config['loss_weights']['ObstacleFreePathLoss'] * loss_obstacle_avoidance + self.config['loss_weights']['GraphBasedConsistencyLoss'] * loss_graph_based_consistency

        ################################################
        ### GENERATE RAMDOM PATHS FOR THE SAME START AND END###
        ################################################
        indx = np.random.randint(0, x_path.shape[0])
        batch_size = x_path.shape[0]
        # take only the 1st path
        first_path = batch[self.path_type][indx]
        first_world_cond = batch[self.cond_type][indx]
        x = first_path.unsqueeze(0).expand(batch_size, -1, -1)
        world_cond = first_world_cond.unsqueeze(0).expand(batch_size, -1, -1, -1)
        start_pos = x[:, 0,:]
        end_pos = x[:, -1,:]
        start_and_end_pos = x[:, [0, -1],:]
        # encode conditionings
        # encoder world
        world_encoding = self.world_encoder(world_cond)
        # start position encoding
        start_end_encoding = self.start_end_encoder(start_and_end_pos)
        # fuse the conditionings world encoding + start encoding + end encoding
        # cond = world_encoding + start_end_encoding[:, 0, :] + start_end_encoding[:, 1, :]
        cond = world_encoding
        # diffuse entire path
        samples_offset, intermediates_offset = self.diffusion.sample(self.model, n=x_path.shape[0], cond=cond)
        intermediates_for_same_point = torch.stack([encoded_path_to_real_path(encoded_path=intermediate_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos) for intermediate_offset in intermediates_offset])
        samples_for_same_point = encoded_path_to_real_path(encoded_path=samples_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos)

        ################################################
        ### Inpainting from preset path###
        ################################################
        constraints = {
            0: torch.zeros_like(x_path[indx, 0,:]),
            # point in the middle as well on the straight path
            x_path.shape[1] // 2: torch.zeros_like(x_path[indx, x_path.shape[1] // 2,:]),
            x_path.shape[1] - 1: torch.zeros_like(x_path[indx, -1,:]),
        }
        # samples_offset, intermediates_offset = self.diffusion.sample_inpainting(model=self.model, n=x_path.shape[0], cond=cond, constraints=constraints)
        # intermediates_inpainting = torch.stack([encoded_path_to_real_path(encoded_path=intermediate_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos) for intermediate_offset in intermediates_offset])
        # samples_inpainting = encoded_path_to_real_path(encoded_path=samples_offset, straight_path=x_straigh_path, path_type=self.path_type, start=start_pos, end=end_pos)

        # log
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_loss/obstacle_avoidance', loss_obstacle_avoidance)
        self.log('val_loss/graph_based_consistency', loss_graph_based_consistency)
        # # optim 
        self.log('optim/learning_rate', self.trainer.optimizers[0].param_groups[0]['lr'])
        self.logging_cache = {
            'samples': samples,
            'intermediates': intermediates,
            'x_path': x_path,
            'x_world_img': x_world_img,
            'x_world_img_distance_field': x_world_img_distance_field,
            'x_offset_path': x_offset_path,
            # ## 
            # 'samples_for_same_point':samples_for_same_point,
            # 'intermediates_for_same_point':intermediates_for_same_point,
            # 'x_path_for_same_point':x_path[indx].unsqueeze(0).expand(batch_size, -1, -1),
            # 'x_world_img_for_same_point':x_world_img[indx].unsqueeze(0).expand(batch_size, -1, -1),
            # 'x_offset_path_for_same_point':x_offset_path[indx].unsqueeze(0).expand(batch_size, -1, -1),
            # ###
            # 'samples_inpainting': samples_inpainting,
            # 'intermediates_inpainting': intermediates_inpainting,
            # 'x_path_inpainting': x_path,
            # 'x_world_img_inpainting': x_world_img,
            # 'x_world_img_distance_field_inpainting': x_world_img_distance_field,
            # 'x_offset_path_inpainting': x_offset_path,
            # 'constraints': constraints,
        }

        return {'val_loss': loss}
