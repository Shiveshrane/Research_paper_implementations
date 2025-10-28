import torch
import torch.nn.functional as F
import sentencepiece as spm
import numpy as np
import time
from typing import Dict, Optional


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps,
        device=mask_index.device,
        dtype=torch.int64
    ) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt_ids, tokenizer, steps=64, gen_length=128,
             block_length=128, temperature=1.0, cfg_scale=0.,
             remasking='low_confidence', mask_id=6,
             logits_eos_inf=False, confidence_eos_eot_inf=False,
             return_timing=False):
    timing = {
        'total': 0.0,
        'model_forward': 0.0,
        'sampling': 0.0,
        'remasking': 0.0,
        'per_step': []
    }

    start_time = time.time()

    device = next(model.parameters()).device

    if len(prompt_ids) > 0:
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    else:
        prompt = torch.tensor([[]], dtype=torch.long, device=device)

    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device
    )

    if prompt.shape[1] > 0:
        x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0, \
        f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0, \
        f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
    steps_per_block = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)

        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            step_start = time.time()

            mask_index = (x == mask_id)

            forward_start = time.time()
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)

                logits = model(x_)
                logits, un_logits = torch.chunk(logits, 2, dim=0)

                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            timing['model_forward'] += time.time() - forward_start

            sampling_start = time.time()

            if logits_eos_inf:
                eos_id = tokenizer.piece_to_id('[EOS]')
                if eos_id >= 0:
                    logits[:, :, eos_id] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)

            x0 = torch.argmax(logits_with_noise, dim=-1)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            timing['sampling'] += time.time() - sampling_start

            remasking_start = time.time()

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)

                if confidence_eos_eot_inf:
                    eos_id = tokenizer.piece_to_id('[EOS]')
                    if eos_id >= 0:
                        p[:, :, eos_id] = 0

                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                )

            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(f"Remasking strategy '{remasking}' not implemented")

            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                if num_transfer_tokens[j, i] > 0:
                    _, select_index = torch.topk(
                        confidence[j],
                        k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            timing['remasking'] += time.time() - remasking_start

            step_time = time.time() - step_start
            timing['per_step'].append(step_time)

    timing['total'] = time.time() - start_time

    if return_timing:
        return x, timing
    return x


class LLaDAInference:
    def __init__(self, model, tokenizer_path, device='cuda'):
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        self.sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
        self.mask_id = self.sp.piece_to_id('[MASK]')

        print(f"Initialized LLaDA Inference")
        print(f"  Mask token ID: {self.mask_id}")
        print(f"  Vocab size: {self.sp.get_piece_size()}")

    def generate_text(self, prompt="", max_length=256, steps=64,
                     temperature=1.0, cfg_scale=0.,
                     remasking='low_confidence', block_length=None,
                     verbose=True):
        overall_start = time.time()

        encode_start = time.time()
        if prompt:
            prompt_ids = self.sp.encode_as_ids(prompt)
        else:
            prompt_ids = []
        encode_time = time.time() - encode_start

        if block_length is None:
            block_length = max_length

        output_ids, timing = generate(
            model=self.model,
            prompt_ids=prompt_ids,
            tokenizer=self.sp,
            steps=steps,
            gen_length=max_length,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking=remasking,
            mask_id=self.mask_id,
            logits_eos_inf=False,
            confidence_eos_eot_inf=False,
            return_timing=True
        )

        decode_start = time.time()
        output_ids = output_ids[0].cpu().tolist()

        filtered_ids = [
            tid for tid in output_ids
            if tid not in [self.mask_id, 0, 1, 2, 3]
        ]

        generated_text = self.sp.decode(filtered_ids)
        decode_time = time.time() - decode_start

        timing['encoding'] = encode_time
        timing['decoding'] = decode_time
        timing['overall'] = time.time() - overall_start

        num_tokens = len(filtered_ids) - len(prompt_ids)
        timing['tokens_per_second'] = num_tokens / timing['total'] if timing['total'] > 0 else 0
        timing['ms_per_token'] = (timing['total'] * 1000) / num_tokens if num_tokens > 0 else 0
        timing['num_tokens'] = num_tokens
        timing['num_steps'] = len(timing['per_step'])
        timing['avg_step_time'] = np.mean(timing['per_step']) if timing['per_step'] else 0

        if verbose:
            self._print_timing(timing, prompt, generated_text)

        return generated_text, timing

    def _print_timing(self, timing: Dict, prompt: str, generated_text: str):
        print("\n" + "="*80)
        print("GENERATION TIMING REPORT")
        print("="*80)

        if prompt:
            print(f"Prompt: {prompt[:50]}{'...' if len(prompt) > 50 else ''}")
        print(f"Generated tokens: {timing['num_tokens']}")
        print(f"Denoising steps: {timing['num_steps']}")
        print()

        print("TIMING BREAKDOWN:")
        print(f"  Encoding:       {timing['encoding']*1000:>8.2f} ms")
        print(f"  Generation:     {timing['total']*1000:>8.2f} ms")
        print(f"    - Forward:    {timing['model_forward']*1000:>8.2f} ms ({timing['model_forward']/timing['total']*100:.1f}%)")
        print(f"    - Sampling:   {timing['sampling']*1000:>8.2f} ms ({timing['sampling']/timing['total']*100:.1f}%)")
        print(f"    - Remasking:  {timing['remasking']*1000:>8.2f} ms ({timing['remasking']/timing['total']*100:.1f}%)")
        print(f"  Decoding:       {timing['decoding']*1000:>8.2f} ms")
        print(f"  Total:          {timing['overall']*1000:>8.2f} ms")
        print()

        print("PERFORMANCE:")
        print(f"  Tokens/second:  {timing['tokens_per_second']:>8.2f} tok/s")
        print(f"  Ms/token:       {timing['ms_per_token']:>8.2f} ms/tok")
        print(f"  Avg step time:  {timing['avg_step_time']*1000:>8.2f} ms/step")
        print("="*80)
        print()

    def infill(self, text_before, text_after, infill_length=20, steps=32,
               temperature=1.0, verbose=True):
        timing = {
            'total': 0.0,
            'model_forward': 0.0,
            'sampling': 0.0,
            'remasking': 0.0,
            'per_step': []
        }

        start_time = time.time()

        before_ids = self.sp.encode_as_ids(text_before)
        after_ids = self.sp.encode_as_ids(text_after)

        total_length = len(before_ids) + infill_length + len(after_ids)
        x = torch.full((1, total_length), self.mask_id, dtype=torch.long, device=self.device)

        x[0, :len(before_ids)] = torch.tensor(before_ids, device=self.device)
        x[0, -len(after_ids):] = torch.tensor(after_ids, device=self.device)

        prompt_index = (x != self.mask_id)
        mask_index = (x == self.mask_id)

        num_transfer_tokens = get_num_transfer_tokens(mask_index, steps)

        for i in range(steps):
            step_start = time.time()
            current_mask = (x == self.mask_id)

            forward_start = time.time()
            logits = self.model(x)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            timing['model_forward'] += time.time() - forward_start

            sampling_start = time.time()
            logits_with_noise = add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            timing['sampling'] += time.time() - sampling_start

            remasking_start = time.time()
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(
                torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
            )

            x0 = torch.where(current_mask, x0, x)
            confidence = torch.where(current_mask, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            _, select_index = torch.topk(confidence[0], k=num_transfer_tokens[0, i])
            transfer_index[0, select_index] = True

            x[transfer_index] = x0[transfer_index]

            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            timing['remasking'] += time.time() - remasking_start

            step_time = time.time() - step_start
            timing['per_step'].append(step_time)

        timing['total'] = time.time() - start_time

        output_ids = x[0].cpu().tolist()
        filtered_ids = [tid for tid in output_ids if tid not in [self.mask_id, 0, 1, 2, 3]]
        generated_text = self.sp.decode(filtered_ids)

        timing['num_tokens'] = infill_length
        timing['num_steps'] = len(timing['per_step'])
        timing['avg_step_time'] = np.mean(timing['per_step']) if timing['per_step'] else 0
        timing['tokens_per_second'] = infill_length / timing['total'] if timing['total'] > 0 else 0
        timing['ms_per_token'] = (timing['total'] * 1000) / infill_length if infill_length > 0 else 0

        if verbose:
            print("\n" + "="*80)
            print("INFILLING TIMING REPORT")
            print("="*80)
            print(f"Infilled tokens: {infill_length}")
            print(f"Denoising steps: {timing['num_steps']}")
            print()
            print("TIMING:")
            print(f"  Total:          {timing['total']*1000:>8.2f} ms")
            print(f"  Tokens/second:  {timing['tokens_per_second']:>8.2f} tok/s")
            print(f"  Ms/token:       {timing['ms_per_token']:>8.2f} ms/tok")
            print("="*80)
            print()

        return generated_text, timing
