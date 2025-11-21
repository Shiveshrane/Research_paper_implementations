import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id, token):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    login(token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    return model, tokenizer, device
