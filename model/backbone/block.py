import torch.nn as nn
from timm.models.layers.helpers import to_2tuple
from ..pet_mixin import AdapterMixin, PrefixMixin, PromptMixin
class Mlp(nn.Module, AdapterMixin):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.adapt_module("fc1", x)  # x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.adapt_module("fc2", x)  # x = self.fc2(x)
        x = self.drop2(x)
        return x


class Attention(nn.Module, PromptMixin, PrefixMixin, AdapterMixin):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias) # Conv2 LoRA 쓸려면 여기를 고쳐야됨.. 근데 사전 학습된거는 다 Linear인데..?
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = self.add_prompt(x)

        B, N, C = x.shape
        qkv = self.adapt_module("qkv", x)

        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv.chunk(3, dim=-1)
        k, v = self.add_prefix(k, v)

        q = q.reshape(B, N, self.num_heads, C // self.num_heads)
        q = q.permute(0, 2, 1, 3)

        k = k.reshape(B, -1, self.num_heads, C // self.num_heads)
        k = k.permute(0, 2, 1, 3)

        v = v.reshape(B, -1, self.num_heads, C // self.num_heads)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = self.compensate_prefix(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.adapt_module("proj", x)  # x = self.proj(x)
        x = self.proj_drop(x)

        x = self.reduce_prompt(x)

        return x

