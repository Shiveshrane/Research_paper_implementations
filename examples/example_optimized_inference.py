import torch
from models.llada import LlaDA
from config.model_args import ModelArgs
from inference.optimized_inference import LLaDAInference

def example_usage():
    model = LlaDA(device='cuda')
    checkpoint = torch.load('best_model.pt', map_location='cuda')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    inferencer = LLaDAInference(model, 'tokenizer.model', device='cuda')

    print("="*80)
    print("EXAMPLE 1: Unconditional Generation")
    print("="*80)
    text, timing = inferencer.generate_text(
        prompt="",
        max_length=128,
        steps=64,
        temperature=1.0
    )
    print(text)

    print("\n" + "="*80)
    print("EXAMPLE 2: Conditional Generation (Continue Prompt)")
    print("="*80)
    text, timing = inferencer.generate_text(
        prompt="ROMEO: ",
        max_length=128,
        steps=64,
        temperature=0.9
    )
    print(text)

    print("\n" + "="*80)
    print("EXAMPLE 3: Semi-Autoregressive (Faster)")
    print("="*80)
    text, timing = inferencer.generate_text(
        prompt="To be or not to be",
        max_length=128,
        steps=32,
        block_length=32,
        temperature=1.0
    )
    print(text)

    print("\n" + "="*80)
    print("EXAMPLE 4: Infilling")
    print("="*80)
    text, timing = inferencer.infill(
        text_before="ROMEO: ",
        text_after=" is the question.",
        infill_length=10,
        steps=32
    )
    print(text)

    print("\n" + "="*80)
    print("EXAMPLE 5: Greedy Decoding (temperature=0)")
    print("="*80)
    text, timing = inferencer.generate_text(
        prompt="Once upon a time",
        max_length=64,
        steps=32,
        temperature=0
    )
    print(text)


if __name__ == "__main__":
    example_usage()
