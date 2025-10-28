import torch
from torch.utils.data import DataLoader
from config.model_args import ModelArgs
from models.llada import LlaDA
from utils.dataset import Dataset
from training.train import train

def main():
    text_file_path='input.txt'
    tokenizer_path='tokenizer.model'
    train_dataset=Dataset(text_file_path, tokenizer_path, split="train")
    val_dataset=Dataset(text_file_path, tokenizer_path, split="val")

    train_dataloader=DataLoader(
        train_dataset,
        batch_size=ModelArgs.batch_size,
        shuffle=True
    )
    val_dataloader=DataLoader(
        val_dataset,
        batch_size=ModelArgs.batch_size,
    )

    trained_model = train(train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
