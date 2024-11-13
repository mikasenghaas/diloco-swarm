# Adapted from: https://github.com/karpathy/nanoGPT/blob/master/model.py

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.nn import functional as F

from huggingface_hub import PyTorchModelHubMixin

@dataclass
class GPT2Config():
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 Tokenizer (padded to nearest multiple of 64)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # (Like GPT-2, but better without)

class LayerNorm(nn.Module):
    def __init__(self, dim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, dropout: float, block_size: int):
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd {n_embd} must be divisible by n_head {n_head}"
        
        # Save parameters
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

        # Linear layers
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Flash attention (optional)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (hidden_dim)

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, n_embd: int, bias: bool, dropout: float):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))

class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, bias: bool, dropout: float, block_size: int):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = CausalSelfAttention(n_embd, n_head, bias, dropout, block_size)
        self.ln_2 = LayerNorm(n_embd, bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: GPT2Config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([TransformerBlock(config.n_embd, config.n_head, config.bias, config.dropout, config.block_size) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def num_parameters(self, non_embedding: bool = True) -> int:
        num_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            num_params -= self.transformer.wpe.weight.numel()
        return num_params

    def init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        device = input_ids.device

        # Position embeddings
        pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=device)
        pos_emb = self.transformer.wpe(pos)

        # Token embeddings
        tok_emb = self.transformer.wte(input_ids)

        # Forward the Transformer model itself
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @classmethod
    def from_hf(cls, hf_model, override_args=None):
        model_type = hf_model.config._name_or_path.split("/")[-1]
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}
        assert all(k == 'dropout' for k in override_args)

        # Get the config from the model type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        if 'dropout' in override_args:
            config_args['dropout'] = override_args['dropout']

        # Create model
        config = GPT2Config(**config_args)
        model = GPT2(config)
        
        # Get all relevant keys from state dict
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # Get all relevant keys from the Hugging Face model
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_type)
        hf_sd = hf_model.state_dict()

        # Copy parameters
        hf_sd_keys = hf_sd.keys()
        hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        hf_sd_keys = [k for k in hf_sd_keys if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        assert len(hf_sd_keys) == len(sd_keys), f"mismatched keys: {len(hf_sd_keys)} != {len(sd_keys)}"
        for k in hf_sd_keys:
            if any(k.endswith(w) for w in transposed): # Uses some 1D convolutions which we need to transpose
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])

        return model

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, num_samples: int, max_new_tokens: int, temperature: float = 1.0, top_k: Optional[int] = None, eos_token_id: Optional[int] = None, **kwargs):
        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.config.block_size else input_ids[:, -self.config.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return input_ids