import torch
from config.model_args import ModelArgs

def forward_process(batch, total_dim=ModelArgs.forward_dim, eps=ModelArgs.forward_eps):
  b,l=batch.shape
  t=torch.rand((b,), device=batch.device)
  p_mask=(1-eps)*t+eps
  p_mask=p_mask[:, None].repeat(1,l)
  mask_indices=torch.rand((b,l), device=batch.device)<p_mask
  noisy_batch=torch.where(mask_indices, total_dim, batch)
  return noisy_batch, mask_indices, p_mask
