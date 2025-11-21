import torch

def get_all_activations(model, tokenizer, prompts, device):
  activations={}
  for i, prompt in enumerate(prompts):
    model_inputs=tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
      outputs = model(**model_inputs, output_hidden_states=True)
      for j, output in enumerate(outputs.hidden_states):
        last_tok_vec=output[0,-1,:].cpu()
        if j not in activations:
          activations[j]=[]
        activations[j].append(last_tok_vec)

  final_means={}
  for layer_id, layer_activations in activations.items():
    final_means[layer_id]=torch.stack(layer_activations).mean(dim=0)
  return final_means

def get_mean_activations(model, tokenizer, prompts, device, layer_id):
  activations=[]
  for p in prompts:
    input_ids = tokenizer(p, return_tensors="pt").to(device)
    with torch.no_grad():
      outputs = model(**input_ids,output_hidden_states=True)
      output=outputs.hidden_states[layer_id]
      last_token_vec=output[0,-1,:]
      activations.append(last_token_vec.cpu())

  return torch.stack(activations).mean(dim=0)
