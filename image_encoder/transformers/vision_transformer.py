# Code borrowed from DINOv2
# https://github.com/facebookresearch/dinov2/tree/main
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, temperature):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # Store q, k, v as one matrix and unpack on fwd pass
        q, k, v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale) / temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.0, bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, temperature):
        x_temp, attn = self.attn(self.norm1(x), temperature)
        x = x + x_temp
        x_temp = self.mlp(self.norm2(x))
        x = x + x_temp
        return x, attn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        max_img_size=32,
        unique_patches=13,
        embed_dim=32,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        head=False,
        output_classes=13,
    ):
        """
        Args:
            max_img_size (int): max input image size
            unique_patches (int): unique ints in input grids
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            head (bool): if true, add final linear layer to output logits for classification
            output_classes (int): output dimension of your classification head
        """
        super().__init__()
        self.model_params = {
            key: value for key, value in locals().items()
            if key in ["max_img_size", "unique_patches", "embed_dim", "depth", "num_heads", "mlp_ratio", "qkv_bias", "drop_rate", "attn_drop_rate"]
        }   # Store model params for easy reloading
        
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(embed_dim)
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.head = head

        self.patch_embeddings = nn.Embedding(num_embeddings=unique_patches+1, embedding_dim=embed_dim)
        self.mask_token = unique_patches    # Token id for the masking token
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.register_buffer('pos_enc', self.get_positional_encodings(max_img_size, embed_dim))
        
        self.blocks = nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate
                )
                for i in range(depth)
            ]
        )
        if self.head:
            self.classification_head = nn.Linear(embed_dim, output_classes, bias=False)
        
        print(f"Vision Transformer instantiated with {sum(p.numel() for p in self.parameters() if p.requires_grad):,} parameters.")

    def forward(self, x, mask=None, save_attn=False, temperature=1):
        B, H, W = x.shape
        if mask is not None:
            x = torch.where(mask == 1, self.mask_token, x)
        x = self.pos_enc[:, :H, :W] + self.patch_embeddings(x)
        x = x.view(B, H * W, -1)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        attns = []
        for blk in self.blocks:
            x, attn = blk(x, temperature)
            if save_attn:
                attns.append(attn)
        x = self.norm(x)    # B x N x C
        
        cls_logits = x[:, :1, :]         # B x 1 x C   # Just the class token
        patch_logits = x[:, 1:, :]       # B x (N-1) x C    # Just the patch values
        if self.head:
            patch_logits = self.classification_head(patch_logits)     # B x (N-1) x output_classes
        return cls_logits, patch_logits, attns

    @staticmethod
    def get_positional_encodings(max_img_size, embed_dim):
        """ 
        Given max_img_size, computes positional encodings using a sinusoidal method on two channels and returns a tensor of size 'max_img_size x max_img_size x embed_dim'
        """
        def _positional_encoding_2d(x, y, d):
            x, y = torch.tensor(float(x)), torch.tensor(float(y))            
            pe_x = torch.tensor([torch.sin(x / (10000 ** (2 * (k // 2) / d))) if k % 2 == 0 else torch.cos(x / (10000 ** (2 * (k // 2) / d)))
                                 for k in range(d // 2)])
            pe_y = torch.tensor([torch.sin(y / (10000 ** (2 * (k // 2) / d))) if k % 2 == 0 else torch.cos(y / (10000 ** (2 * (k // 2) / d)))
                                 for k in range(d // 2)])
            pe = torch.cat([pe_x, pe_y])
            return pe

        positional_grid = torch.zeros((max_img_size, max_img_size, embed_dim))
        for x in range(max_img_size):
            for y in range(max_img_size):
                positional_grid[x, y, :] = _positional_encoding_2d(x, y, embed_dim)        
        return positional_grid

    def save_model(self):
        save_dir = "trained_models"
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"vit_{timestamp}.pth"
        save_path = os.path.join(save_dir, save_name)
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_params': self.model_params
        }, save_path)
        print(f"Model and parameters saved to {save_path}")

    @classmethod
    def load_model(cls, path):
        checkpoint = torch.load(path)
        model = cls(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
        return model