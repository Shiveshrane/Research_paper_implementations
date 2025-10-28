import torch
import numpy as np
import sentencepiece as spm
from config.model_args import ModelArgs

class Dataset(torch.utils.data.Dataset):
  def __init__(self, text_file_path, tokenizer_path, ctx_len=ModelArgs.max_seq_len, split="train",val_split=0.1):
    self.text_file=text_file_path
    self.ctx_len=ctx_len
    self.sp=spm.SentencePieceProcessor(model_file=tokenizer_path)
    self.split=split
    self.val_split=val_split
    with open(self.text_file, 'r', encoding='utf-8') as f:
      self.data=f.read()
    all_tokens=self.sp.encode_as_ids(self.data)
    n=len(all_tokens)
    train_len=int(n*(1-self.val_split))
    if self.split=="train":
      self.data_np=np.array(all_tokens[:train_len], dtype=np.int64)
    else:
      self.data_np=np.array(all_tokens[train_len:],dtype=np.int64)

    self.data=torch.from_numpy(self.data_np)

    self._n=max(0, len(self.data)-self.ctx_len)
  def __len__(self):
    return self._n

  def __getitem__(self, idx):
    return self.data[idx:idx+self.ctx_len]
