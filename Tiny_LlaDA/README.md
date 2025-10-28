# LlaDA - Masked Diffusion Language Model
This tiny-LlaDA is a simplified version of https://github.com/ML-GSAI/LLaDA. 

Paper: https://arxiv.org/pdf/2502.09992

This model is trained on https://huggingface.co/datasets/karpathy/tiny_shakespeare for 10 epochs on Kaggle using 1 x Tesla T4 GPU. 

## Project Structure

```
LlaDA/
├── config/
│   └── model_args.py          # Model configuration and hyperparameters
├── models/
│   ├── rope.py                # Rotary Position Embedding
│   ├── normalization.py       # RMS Normalization
│   ├── attention.py           # Multi-Head Bidirectional Attention
│   ├── mlp.py                 # MLP with SwiGLU activation
│   ├── encoder.py             # Encoder layer
│   └── llada.py               # Main LlaDA model
├── utils/
│   ├── dataset.py             # Dataset class
│   ├── forward_process.py     # Forward diffusion process
│   └── lr_scheduler.py        # Learning rate scheduler
├── training/
│   └── train.py               # Training loop
├── inference/
│   ├── basic_inference.py     # Basic inference implementation
│   └── optimized_inference.py # Optimized inference with timing
├── examples/
│   ├── example_basic_inference.py
│   └── example_optimized_inference.py
└── main.py                    # Main training script
```

## Usage

### Training
```python
python main.py
```

### Basic Inference
```python
python examples/example_basic_inference.py
```

### Optimized Inference
```python
python examples/example_optimized_inference.py
```
## Training code reference (As stated in the paper and the official Github Repo):
https://github.com/ML-GSAI/SMDM
