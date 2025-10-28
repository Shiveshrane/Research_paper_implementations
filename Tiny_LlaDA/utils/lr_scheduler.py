def get_lr_scheduler(peak_lr, train_epoch, steps_per_epoch, min_lr=2e-5, warmup_frac=0.1, decay_frac=0.2):
  total_iters=train_epoch*steps_per_epoch
  warmup_iters=int(total_iters*warmup_frac)
  decay_iters=int(total_iters*(1-decay_frac))
  
  def get_lr(it):
    if it<warmup_iters:
      return peak_lr*it/warmup_iters
    if it>warmup_iters and it<decay_iters:
      return peak_lr
    if it>=total_iters or decay_iters>=total_iters:
      return min_lr

    iters_into_decay=it-decay_iters
    decay_duration=total_iters-decay_iters
    decay_ratio=iters_into_decay/decay_duration
    decay_ratio=max(0.0, min(1.0, decay_ratio))
    return peak_lr-(peak_lr-min_lr)*decay_ratio
  
  return get_lr
