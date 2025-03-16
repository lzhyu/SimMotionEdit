from src.model.DiT_models import *

class EncoderBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x, c, mask=None):
        x = x + self.attn((self.norm1(x)), mask)
        x = x + self.mlp((self.norm2(x)))
        return x
    
    def forward_with_att(self, x, c, mask=None):
        attention_out, attention_mask = self.attn.forward_w_attention((self.norm1(x)), mask)
        x = x + attention_out
        x = x + self.mlp((self.norm2(x)))
        return x, attention_mask

import copy
def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        # features = d_model
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
    
class TransEncoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N: int):
        super(TransEncoder, self).__init__()
        self.layers = clones(layer, N)

    def forward(self, x: torch.FloatTensor, mask: torch.ByteTensor) -> torch.FloatTensor:
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return x
    def forward_with_att(self, x, c, mask=None):
        attention_masks = []
        for layer in self.layers:
            x, attention_mask = layer.forward_with_att(x, c, mask)
            attention_masks.append(attention_mask)
        return x, attention_masks