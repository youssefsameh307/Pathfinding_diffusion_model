# Add import 
import sys
import torch 
from torch import nn
from torch import optim
from prodigyopt import Prodigy # proddigy optimizer from https://github.com/konstmish/prodigy?tab=readme-ov-file
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset.RobotPathDataset.normalizer import MinMaxFeatureNormalizer

from model.models import EMA
import copy
import time

# remove all warnings
import warnings
from dataset import RobotPathDataset
from model.model_blocks import Sine
from model.model_blocks import PositionalEncoding

class TransformerEncoderConditionalLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feed_forward_ratio=4):
        super(TransformerEncoderConditionalLayer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feed_forward_ratio*d_model, batch_first=True)
        self.adaLN_modulation = nn.Sequential( # predicts for each channel 3 modulation parameters mean, scale, gate
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )

        self.initialize_weights()
    def initialize_weights(self):
        # initialize the adaLN modulation layer to 0
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond=None):
        # cond shape is (batch, d_model)
        modulation_values = self.adaLN_modulation(cond) # (batch, 3*d_model)
        shift_msa, scale_msa, gate_msa = modulation_values.chunk(3, dim=1) # ()
        modulated = modulate(x, shift_msa, scale_msa)
        x = self.layer(modulated)
        gated = gate_msa.unsqueeze(1) * x
        return gated

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransformerEncoderConditionalLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feed_forward_ratio=4):
        super(TransformerEncoderConditionalLayer, self).__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feed_forward_ratio*d_model, batch_first=True)
        self.adaLN_modulation = nn.Sequential( # predicts for each channel 3 modulation parameters mean, scale, gate
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )

        self.initialize_weights()
    def initialize_weights(self):
        # initialize the adaLN modulation layer to 0
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, cond=None):
        # cond shape is (batch, d_model)
        modulation_values = self.adaLN_modulation(cond) # (batch, 3*d_model)
        shift_msa, scale_msa, gate_msa = modulation_values.chunk(3, dim=1) # ()
        modulated = modulate(x, shift_msa, scale_msa)
        x = self.layer(modulated)
        gated = gate_msa.unsqueeze(1) * x
        return gated

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, cond_dim ,d_model, nhead, num_layers, dim_feed_forward_ratio=4, depth=1, max_len=264,use_sin_activation=False, use_adaln=True):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.depth = depth
        # Conditional
        self.timestep_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            Sine() if use_sin_activation else nn.ReLU(),
        )
        
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            Sine() if use_sin_activation else nn.ReLU(),
        )
        
        self.cond_norm = nn.LayerNorm(d_model) # replace with ada norm from https://github.com/facebookresearch/DiT/tree/main
        
        # map the input to the d_model
        self.process_input = nn.Sequential(
            nn.Linear(input_dim, d_model),
            Sine() if use_sin_activation else nn.ReLU(),
        )


        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=max_len)

        self.block = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feed_forward_ratio*d_model, batch_first=True) if not use_adaln else TransformerEncoderConditionalLayer(d_model, nhead, dim_feed_forward_ratio=dim_feed_forward_ratio)

        self.blocks = nn.ModuleList([
            copy.deepcopy(self.block) for _ in range(depth)
        ])
        
        self.proces_output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
            # No activation layer as output is gusassian noise anyway
        )

        self.cache = {} # used for logs and debugging
    def forward(self, x, t,cond=None):
        # x shape is (batch, n_waypoints, transtion_dim=2)
        # cond shape is (batch, cond_dim)
        t = t.unsqueeze(1) # (batch) to (batch, 1)
        emb = self.timestep_encoder(t.type(torch.float32)) # maps timestep to d_model (batch, d_model)
        self.cache['emb_timestep'] = emb
        if cond is not None:
            cond = self.cond_encoder(cond) # maps cond_dim to d_model (batch, d_model)
            self.cache['emb_cond_after_dense'] = cond
            emb = emb + cond # (batch, d_model)
        emb = self.cond_norm(emb) # (batch, d_model)
        self.cache['emb_normalized'] = emb
        
        
        x = self.process_input(x) # maps transtion_dim to d_model
        x = self.positional_encoding(x) # add positional encoding
        for i in range(self.depth):
            x = self.blocks[i](x, emb)
            self.cache['block_{}-adaLN'.format(i)] = self.block.adaLN_modulation[-1].weight 

        # map the output to the output_dim
        x = self.proces_output(x)     
        return x
