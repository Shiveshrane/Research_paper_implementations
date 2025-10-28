import torch

class ModelArgs:
  base_freq=10000.0
  batch_size=32
  embed_dims=512
  max_seq_len=256
  num_heads=8
  attn_dropout=0.2
  eps=1e-6
  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
  num_blocks=8
  dropout=0.2
  vocab_size=33488
  forward_dim=6
  forward_eps=1e-3
  AdamW_weight_decay=1e-1
  decay_lr=True
  beta1=0.9
  beta2=0.95
