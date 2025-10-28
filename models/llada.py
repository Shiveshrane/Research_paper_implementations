import torch.nn as nn
from config.model_args import ModelArgs
from models.encoder import Encoder_layer
from models.normalization import RMSNorm

class LlaDA(nn.Module):
  def __init__(self, embed_dims=ModelArgs.embed_dims, num_heads=ModelArgs.num_heads, attn_dropout=ModelArgs.attn_dropout, num_blocks=ModelArgs.num_blocks,  device=None):
    super().__init__()
    self.embed_dims=embed_dims
    self.num_heads=num_heads
    self.attn_dropout=attn_dropout
    self.num_blocks=num_blocks
    self.device=device
    self.embeddings=nn.Embedding(ModelArgs.vocab_size, self.embed_dims)
    self.layers=nn.ModuleList([Encoder_layer(embed_dims=self.embed_dims, num_heads=self.num_heads, attn_dropout=self.attn_dropout, device=self.device) for _ in range(self.num_blocks)])
    self.norm=RMSNorm(embed_dims=self.embed_dims, device=self.device)
    self.dropout=nn.Dropout(ModelArgs.dropout)
    self.output_layer=nn.Linear(self.embed_dims, ModelArgs.vocab_size, bias=False)

  def forward(self, x):
    b,s=x.shape
    x=self.embeddings(x)
    for layer in self.layers:
      x=layer(x)
    x=self.norm(x)
    x=self.dropout(x)
    x=self.output_layer(x)
    return x
