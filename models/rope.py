import torch
import torch.nn as nn
from config.model_args import ModelArgs

class RoPE(nn.Module):
  def __init__(self, base_freq=ModelArgs.base_freq, embed_dims=ModelArgs.embed_dims, max_seq_len=ModelArgs.max_seq_len):
    super().__init__()
    self.base_freq=base_freq
    self.embed_dims=embed_dims
    self.max_seq_len=max_seq_len
    positions=torch.arange(0, self.max_seq_len, dtype=torch.float)
    theta_num=torch.arange(0,self.embed_dims, 2).float()
    theta=1.0/(self.base_freq**(theta_num/self.embed_dims))
    self.register_buffer("theta", theta)
    angles=positions.unsqueeze(1)*theta.unsqueeze(0)
    self.register_buffer("sine", torch.sin(angles))
    self.register_buffer("cosine", torch.cos(angles))

  def forward(self, x, start_pos=0):
    b,s,h,d=x.shape
    x=x.view(b,s,h,d//2,2)
    cos=self.cosine[start_pos:start_pos+s].unsqueeze(0).unsqueeze(2)
    sin=self.sine[start_pos:start_pos+s].unsqueeze(0).unsqueeze(2)

    x_rot=torch.stack([
        x[...,0]*cos-x[...,1]*sin,
        x[...,0]*sin+x[...,1]*cos
    ], dim=-1)
    return x_rot.view(b,s,h,d)
