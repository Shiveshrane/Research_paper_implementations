import torch
import torch.nn as nn
import torch.nn.functional as F
from config.model_args import ModelArgs
from models.rope import RoPE
from models.normalization import RMSNorm

class MultiHeadBiAttention(nn.Module):
  def __init__(self, embed_dims=ModelArgs.embed_dims, num_heads=ModelArgs.num_heads, attn_dropout=ModelArgs.attn_dropout):
    super().__init__()
    self.embed_dims=embed_dims
    self.num_heads=num_heads
    assert self.embed_dims%self.num_heads==0, "embed_dims must be divisible by num_heads"
    self.attn_dropout=attn_dropout
    self.head_dims=embed_dims//self.num_heads
    self.qk_norm=RMSNorm(self.head_dims)

    self.Wq=nn.Linear(embed_dims, self.head_dims*self.num_heads, bias=False)
    self.Wk=nn.Linear(embed_dims, self.head_dims*self.num_heads, bias=False)
    self.Wv=nn.Linear(embed_dims, self.head_dims*self.num_heads, bias=False)
    self.out=nn.Linear(self.head_dims*self.num_heads, self.embed_dims, bias=False)
    self.rope=RoPE(embed_dims=self.head_dims)
    self.dropout=nn.Dropout(self.attn_dropout)

  def forward(self, x, qk_norm=False):
    B,S,D=x.shape
    q=self.Wq(x)
    k=self.Wk(x)
    v=self.Wv(x)

    if qk_norm:
      q=self.qk_norm(q)
      k=self.qk_norm(k)

    q=q.view(B,S,self.num_heads, self.head_dims)
    k=k.view(B,S,self.num_heads, self.head_dims)
    v=v.view(B,S,self.num_heads, self.head_dims)

    q=self.rope(q)
    k=self.rope(k)

    q=q.transpose(1,2)
    k=k.transpose(1,2)
    v=v.transpose(1,2)

    attn=torch.matmul(q, k.transpose(-2,-1))/(torch.sqrt(torch.tensor(self.head_dims)))
    attn_output=F.softmax(attn,dim=-1)
    attn_output=self.dropout(attn_output)
    attn_output=torch.matmul(attn_output, v)
    attn_output=attn_output.transpose(1,2)
    attn_output=attn_output.contiguous().view(B,S,self.num_heads*self.head_dims)
    attn_output=self.out(attn_output)
    return attn_output
