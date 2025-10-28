import torch
import torch.nn as nn
from config.model_args import ModelArgs

class Swish(nn.Module):
  def __init__(self):
    super().__init__()
    self.sigmoid=nn.Sigmoid()
  def forward(self, x):
    return x*self.sigmoid(x)

class MLP(nn.Module):
  def __init__(self, embed_dims=ModelArgs.embed_dims, device=None):
    super().__init__()
    self.embedding_dims=embed_dims
    self.batch_size=ModelArgs.batch_size
    self.hidden_dims=((self.embedding_dims*2)*4)//3
    self.linear1=nn.Linear(self.embedding_dims, self.hidden_dims, bias=False, device=device)
    self.linear2=nn.Linear(self.embedding_dims, self.hidden_dims, bias=False, device=device)
    self.linear3=nn.Linear(self.hidden_dims, self.embedding_dims, bias=False, device=device)
    self.swish=Swish()

  def forward(self, x):
    x1=self.linear1(x)
    x2=self.linear2(x)
    hidden=torch.mul(x1, self.swish(x2))
    return self.linear3(hidden)
