from torch import nn
import einops
from einops.layers.torch import Rearrange
import pdb
from .model_blocks import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock, Residual, PreNorm, LinearAttention, SinusoidalPosEmbedding2D
import torch
class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5, device='cuda'):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size, device=device),
            Conv1dBlock(out_channels, out_channels, kernel_size, device=device),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels, device=device),
            Rearrange('batch t -> batch t 1')
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1, device=device) \
            if inp_channels != out_channels else nn.Identity(device=device)

        self.to(device)
        self.device = device  
    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class TemporalUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim = 0, # NOT IMPLEMENTED YET
        encoder=None,
        dim=32, # Dimenstion for internal representation of conditioning
        dim_mults=(1, 2, 4),
        attention=False,
        device='cuda'
    ):
        super().__init__()

        self.device = device

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')
        # initialize encoder
        if encoder is not None:
            self.encoder = encoder
            self.encoder.to(device)
        
        # initialize time embedding
        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        
        
        # initialize embedding start pos
        self.start_pos_mlp = nn.Sequential(
            SinusoidalPosEmbedding2D(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # initalize embedding end pos
        self.end_pos_mlp = nn.Sequential(
            SinusoidalPosEmbedding2D(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        horizon_values = [horizon]
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=time_dim, horizon=horizon, device=device),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=time_dim, horizon=horizon, device=device),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2
                horizon_values.append(horizon)

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, device=device)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon, device=device)

        out_in = reversed(in_out[1:])
        for ind, (dim_in, dim_out) in enumerate(out_in):
            
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon, device=device),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=time_dim, horizon=horizon, device=device),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon_values.pop()

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )
        
        self.to(device)



 
    def forward(self, x, time, cond=None, start_pos=None, end_pos=None):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = tim_emb = self.time_mlp(time)
        # Add encoding of the environment to the time embedding
        if cond is not None:
            emb = self.encoder(cond)
            self.last_emb = emb
            # Do this operation on the cpu 
            # t = t.cpu()
            # emb = emb.cpu()
            t = torch.empty(tim_emb.shape, device=self.device)
            t = torch.add(tim_emb, emb) # as doing tim_emb + emb will throw cuda error
            
        if start_pos is not None:
            emb = self.start_pos_mlp(start_pos)
            t = torch.add(t, emb)
        if end_pos is not None:
            emb = self.end_pos_mlp(end_pos)
            t = torch.add(t, emb)

        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x

    def get_last_embedding(self):
        return self.last_emb

class EMA:
    """Exponential Moving Average (EMA) class.

    This class implements the Exponential Moving Average algorithm for model parameter averaging.

    Args:
        beta (float): The smoothing factor for the moving average.

    Attributes:
        beta (float): The smoothing factor for the moving average.
        step (int): The current step count.

    Methods:
        update_model_average: Updates the moving average of the model parameters.
        update_average: Updates the moving average of a single parameter.
        step_ema: Performs a step of the EMA algorithm.
        reset_parameters: Resets the parameters of the EMA model.

    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        """Updates the moving average of the model parameters.

        Args:
            ma_model (nn.Module): The model with the moving average parameters.
            current_model (nn.Module): The model with the current parameters.

        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        """Updates the moving average of a single parameter.

        Args:
            old (torch.Tensor): The old moving average value.
            new (torch.Tensor): The new parameter value.

        Returns:
            torch.Tensor: The updated moving average value.

        """
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        """Performs a step of the EMA algorithm.

        Args:
            ema_model (nn.Module): The model with the moving average parameters.
            model (nn.Module): The model with the current parameters.
            step_start_ema (int, optional): The step at which to start the EMA algorithm. Defaults to 2000.

        """
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        """Resets the parameters of the EMA model.

        Args:
            ema_model (nn.Module): The model with the moving average parameters.
            model (nn.Module): The model with the current parameters.

        """
        ema_model.load_state_dict(model.state_dict())

if __name__=='__main__':
    model = TemporalUnet(horizon=5, transition_dim=32, cond_dim=32)
    print(model.device)

