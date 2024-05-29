# -*- coding: utf-8 -*-

import math
from typing import Optional, Type, Union

import torch
import torch.nn as nn

from ..misc.scaler import Scaler


class Adapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        moe: bool = False,
        kernel_size : int = 3,
        down_sample: Union[float, int] = 5,
        mode: str = "parallel",  # enum before, after, parallel
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(embed_dim * down_sample)

        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            act_layer(),
            nn.Linear(hidden_dim, embed_dim),
            Scaler(scale),
        )
        self.mode = mode
        self.moe = moe
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.layer[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.layer[0].bias)
        nn.init.zeros_(self.layer[2].weight)
        nn.init.zeros_(self.layer[2].bias)

    def forward(self, module, input, **kwargs):
        if not self.moe:
            if self.mode == "before":
                return module(self.layer(input) + input, **kwargs)
            if self.mode == "after":
                return self.layer(module(input, **kwargs)) + input
            return module(input, **kwargs) + self.layer(input)
        else: # moe
            return self.layer(input)


class Conv2dAdapter(nn.Module): # implementation of Convpass
    def __init__(
        self,
        embed_dim: int,
        kernel_size : int = 3,
        down_sample: Union[float, int] = 8,
        mode: str = "parallel",  # enum before, after, parallel
        scale: Optional[float] = None,
        xavier_init = False
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(embed_dim * down_sample)

        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.conv.weight)
        else:
            nn.init.zeros_(self.conv.weight)
            self.conv.weight.data[:, :, 1, 1] += torch.eye(hidden_dim, dtype=torch.float)
        nn.init.zeros_(self.conv.bias)

        self.down = nn.Linear(embed_dim, hidden_dim) # equivalent to 1 * 1 Conv
        self.up = nn.Linear(hidden_dim, embed_dim) # equivalent to 1 * 1 Conv

        self.mode = mode
        self.act = QuickGELU()
        self.scale = Scaler(scale)
        self.dropout = nn.Dropout(0.1)
        self.dim = hidden_dim

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def layer(self, input):
        B, N, C = input.shape # [B, 197, 768]

        input = self.down(input)
        input = self.act(input)

        patch = input[:, 1:].reshape(B, 14, 14, self.dim).permute(0, 3, 1, 2)
        patch = self.conv(patch)
        patch = patch.permute(0, 2, 3, 1).reshape(B, 14*14, self.dim)

        # Class Token도 Conv에 넣어버리네 ㄷㄷ
        cls = input[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        cls = self.conv(cls)
        cls = cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        input = torch.cat([cls, patch], dim=1)

        input = self.act(input)
        input = self.dropout(input)

        input = self.up(input)
        return self.scale(input)

    def forward(self, module, input, **kwargs):
        # input: [1, 197, 768]
        if self.mode == "before":
            input = self.layer(input) + input
            return module(input, **kwargs)
        if self.mode == "after":
            return self.layer(module(input, **kwargs)) + input
        return module(input, **kwargs) + self.layer(input)

class STAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3, # 홀수여야 padding이 들어간다.
        down_sample: Union[float, int] = 5,
        mode: str = 'parallel',
        scale: Optional[float] = None,
        act_layer: Type[nn.Module] = nn.GELU,
    ):
        super().__init__()

        assert mode in ["before", "after", "parallel"], f"Unknown mode {mode}"

        self.hidden_dim = down_sample
        hidden_dim = down_sample
        if isinstance(down_sample, float):
            hidden_dim = int(embed_dim * down_sample)

        self.down = nn.Linear(embed_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=(1,1),
                              padding=((kernel_size - 1) // 2), groups=hidden_dim) # Depth-wise
        self.up = nn.Linear(hidden_dim, embed_dim)

        self.act = act_layer()
        self.scaler = Scaler(scale)
        self.mode = mode
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.down.weight, 0.)
        nn.init.constant_(self.down.bias, 0.)

        nn.init.constant_(self.conv.weight, 0.)
        nn.init.constant_(self.conv.bias, 0.)

        nn.init.constant_(self.up.weight, 0.)
        nn.init.constant_(self.up.bias, 0.)

    def layer(self, input):
        # input: [1, 197, 768]
        b = input.size(0)  # batch_size

        # Empirically, we find that a decent performance can be achieved in case a single
        # ST-Adapter is placed before the MHSA of each transformer block
        id = input.clone()  # id, res: [1, 197, 768]
        res = input.clone()

        input = self.down(input)

        input = input[:, 1:, :].reshape(b, -1, 14, 14)  # class token 날리기, [b, 196, down_sample] -> (B, C, W, H)

        input = self.act(input)
        input = self.conv(input)  # [b, down_sample, 14, 14], Kernel_size = 2 * Padding
        input = input.reshape(b, -1, self.hidden_dim)  # [b, 196, 10]

        input = self.up(input)
        input = self.scaler(input)

        id[:, 1:, :] = input

        return id, res

    def forward(self, module, input, **kwargs):
        id, res = self.layer(input)
        if self.mode == "before":
            return module(id + res , **kwargs) # residual check
        if self.mode == "after":
            return self.layer(module(input, **kwargs))[0] + res
        return module(input, **kwargs) + self.layer(input)[0]

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

