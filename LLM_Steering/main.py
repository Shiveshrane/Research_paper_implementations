import torch
from config import HF_TOKEN, MODEL_ID, positive_prompts, negative_prompts
from model_loader import load_model
from activations import get_all_activations, get_mean_activations
from steering import get_steering_hook

# Setup
model, tokenizer, device = load_model(MODEL_ID, HF_TOKEN)

# Initial test
messages = [
    {"role": "user", "content": "Who are you?"},
]
inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
print(model)

# Get all activations
positive_prompts_activations = get_all_activations(model, tokenizer, positive_prompts, device)
negative_prompts_activations = get_all_activations(model, tokenizer, negative_prompts, device)

steering_vecs={}
layer_mags={}

for layers in positive_prompts_activations.keys():
  steering_vecs[layers]=positive_prompts_activations[layers]-negative_prompts_activations[layers]
  layer_mags[layers]=torch.linalg.norm(steering_vecs[layers]).item()

print(f"{'Layer':<10} | {'Signal Strength':<15}")
print("-" * 30)
for layer_idx, magnitude in layer_mags.items():
    print(f"{layer_idx:<10} | {magnitude:.4f}")

best_layer = max(layer_mags, key=layer_mags.get)
print(f"\nBest Layer appears to be: {best_layer}")

# Best Layer based activation
best_layer=25 # Overwriting as per original script

pos_mean_activations = get_mean_activations(model, tokenizer, positive_prompts, device, best_layer)
neg_mean_activations = get_mean_activations(model, tokenizer, negative_prompts, device, best_layer)

steering_vec=pos_mean_activations-neg_mean_activations
steering_vec=steering_vec.to(device)

steering_vec_norm=torch.linalg.norm(steering_vec).item()

print(steering_vec)
print(steering_vec_norm)

# Steering coeff
steering_coeff=-1.0

# Hook
steering_hook = get_steering_hook(steering_vec, steering_coeff)

prompt="What is 2+2?"

hook_handle=model.model.layers[best_layer].register_forward_hook(steering_hook)

output_ids=tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
  outputs=model.generate(**output_ids,
                         max_new_tokens=40,
                         do_sample=True,
                         temperature=0.9,
                         )
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
hook_handle.remove()
