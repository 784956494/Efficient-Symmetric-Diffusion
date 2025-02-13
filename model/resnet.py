import torch
import torch.nn as nn
from utils.math_utils import SiLU, timestep_embedding
import numpy as np

class ResNet(nn.Module):
    def __init__(self, input_dim, output_dim,
                 hidden_dim=512,
                 time_embed_dim=128,
                 num_res_blocks=4):
        super(ResNet, self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = FCnet(data_dim=input_dim, hidden_dim=hidden_dim, num_res_blocks=num_res_blocks)
        self.norm = nn.GroupNorm(32, hid)
        self.out_module = nn.Sequential(
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, hid),
            SiLU(),
            nn.Linear(hid, output_dim),)

    def forward(self, x, t):
        t_emb = timestep_embedding(t, self.time_embed_dim)
        t_out = self.t_module(t_emb)
        x_out = self.x_module(x)
        out = self.out_module(self.norm(x_out + t_out))
        return out  
    
class FCnet(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype
        self.map = nn.Linear(data_dim, hidden_dim, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hidden_dim, dtype=self.dtype)

        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])

        self.norms = nn.ModuleList(
            [nn.GroupNorm(32, hidden_dim, dtype=self.dtype) for _ in range(num_res_blocks)])

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths = [hid]*4
        for i in range(len(widths) - 1):
            layers.append(nn.Linear(widths[i], widths[i+1], dtype=self.dtype))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.map(x)
        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / np.sqrt(2)
        return h