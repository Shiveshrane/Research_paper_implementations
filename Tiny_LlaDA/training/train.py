import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import tqdm
from config.model_args import ModelArgs
from models.llada import LlaDA
from utils.forward_process import forward_process
from utils.lr_scheduler import get_lr_scheduler

def train(train_dataloader, val_dataloader, peak_lr=2e-4, train_epoch=10, eval_iters=100):
  print("INITIALISING MODEL......")
  model=LlaDA(
      embed_dims=ModelArgs.embed_dims,
      num_heads=ModelArgs.num_heads,
      attn_dropout=ModelArgs.attn_dropout,
      num_blocks=ModelArgs.num_blocks,
      device=ModelArgs.device
  )
  model.to(ModelArgs.device)
  optimizer=torch.optim.AdamW(
      model.parameters(),
      lr=peak_lr,
      betas=(ModelArgs.beta1, ModelArgs.beta2),
      weight_decay=ModelArgs.AdamW_weight_decay
  )
  @torch.no_grad()
  def validate(model, val_dataloader):
    print("validating")
    model.eval()
    losses=torch.zeros(eval_iters, device=ModelArgs.device)
    for k, val_data in enumerate(val_dataloader):
      print(f"iter{k}")
      val_data=val_data.to(ModelArgs.device)
      if k>=eval_iters:
        break
      mc_loss=torch.zeros(16, device=ModelArgs.device)
      for i in range(16):
        input_ids=val_data[:, 0:ModelArgs.max_seq_len].contiguous()
        noisy_input_ids, mask_indices, p_mask=forward_process(input_ids)
        logits=model(noisy_input_ids)
        loss=F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none')
        loss=loss.sum()/(input_ids.shape[0]*input_ids.shape[1])
        mc_loss[i]=loss
        print(loss)
      losses[k]=mc_loss.mean().item()
    out=losses.mean()
    model.train()
    return out

  loss_func=nn.CrossEntropyLoss(reduction=None)
  model.train()

  steps_per_epoch=len(train_dataloader)
  get_lr=get_lr_scheduler(peak_lr, train_epoch, steps_per_epoch)

  global_step=0
  best_val_loss=float("inf")
  for epoch in range(train_epoch):
    print(f"epoch: {epoch}")
    epoch_loss=0.0
    num_batches=0
    start_time=time.time()
    val_loss=validate(model, val_dataloader)
    print(f"val_loss: {val_loss}")
    if val_loss<best_val_loss:
        best_val_loss=val_loss
        torch.save(model.state_dict(), "best_model.pt")

    pbar=tqdm.tqdm(train_dataloader, desc=f"Training Epoch:{epoch+1}/{train_epoch}")
    for batch_idx, batch in enumerate(pbar):
      batch=batch.to(ModelArgs.device)
      lr=get_lr(global_step)
      for param_group in optimizer.param_groups:
        param_group['lr']=lr
      input_ids=batch[:, 0:ModelArgs.max_seq_len].contiguous()
      noisy_input_ids, mask_indices, p_mask=forward_process(input_ids)
      logits=model(noisy_input_ids)
      loss=F.cross_entropy(logits[mask_indices], input_ids[mask_indices], reduction='none')
      loss=loss.sum()/(input_ids.shape[0]*input_ids.shape[1])
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()
      epoch_loss+=loss.item()
      num_batches+=1
      global_step+=1
      pbar.set_postfix({'loss': loss.item(),
                        'lr': lr,
                        'epoch_loss': epoch_loss/num_batches,
                        'time': time.time()-start_time
                        })
  print("\n" + "="*60)
  print("Training Complete! Running final validation...")
  final_val_loss = validate(model, val_dataloader)
  print(f"Final Validation Loss: {final_val_loss:.4f}")
  print(f"Best Validation Loss: {best_val_loss:.4f}")

  torch.save({
      'epoch': train_epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'val_loss': final_val_loss,
      'best_val_loss': best_val_loss,
      'global_step': global_step
  }, "final_model.pt")
  print("Saved final model to 'final_model.pt'")
  print("="*60)

  return model
