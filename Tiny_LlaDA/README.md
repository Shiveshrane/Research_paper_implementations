# LlaDA - Masked Diffusion Language Model

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
