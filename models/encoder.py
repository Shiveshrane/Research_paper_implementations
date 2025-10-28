import torch.nn as nn
from config.model_args import ModelArgs
from models.attention import MultiHeadBiAttention
from models.mlp import MLP
from models.normalization import RMSNorm

class Encoder_layer(nn.Module):
  def __init__(self, embed_dims=ModelArgs.embed_dims, num_heads=ModelArgs.num_heads, attn_dropout=ModelArgs.attn_dropout,dropout=ModelArgs.dropout, device=None):
    super().__init__()
    self.embed_dims=embed_dims
    self.num_heads=num_heads
    self.attn_dropout=attn_dropout
    self.dropout=dropout
    self.attn=MultiHeadBiAttention(embed_dims=self.embed_dims, num_heads=self.num_heads, attn_dropout=self.attn_dropout)
    self.mlp=MLP(embed_dims=self.embed_dims, device=device)
    self.norm1=RMSNorm(embed_dims=self.embed_dims, device=device)
    self.norm2=RMSNorm(embed_dims=self.embed_dims, device=device)
    self.dropout_1=nn.Dropout(self.dropout)
    self.dropout_2=nn.Dropout(self.dropout)
  def forward(self, x):
    b,s,d=x.shape
    x=self.norm1(x)
    x=x+self.dropout_1(self.attn(x))
    x=self.norm2(x)
    x=x+self.dropout_2(self.mlp(x))
    return x
