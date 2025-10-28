import torch
import torch.nn as nn
from config.model_args import ModelArgs

class RMSNorm(nn.Module):
  def __init__(self, embed_dims=ModelArgs.embed_dims, eps=ModelArgs.eps, device=None):
    super().__init__()
    self.eps=eps
    self.embed_dims=embed_dims
    self.gemma=nn.Parameter(torch.ones(self.embed_dims, device=device))

  def norm(self, x):
    val=x.pow(2).mean(-1, keepdim=True)
    val=torch.sqrt(val+self.eps)
    return x/val

  def forward(self, x):
    return self.gemma*self.norm(x)
