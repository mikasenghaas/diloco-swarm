import torch
import torch.nn as nn

from enum import Enum
from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
from src.world import World

class ModelType(Enum):
    LLAMA = "llama"
    GPT = "gpt"

class ShardedModel(nn.Module, ABC):
    def __init__(self, model: AutoModelForCausalLM, world: World):
        super().__init__()
        self.world = world
        self._setup_model_components(model)
        del model

    @abstractmethod
    def _setup_model_components(self, model):
        pass

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // self.world.world_size + (1 if i < num_layers % self.world.world_size else 0) for i in range(self.world.world_size)]
        start_layer = sum(layers_per_gpu[:self.world.local_rank])
        return list(range(start_layer, start_layer + layers_per_gpu[self.world.local_rank]))

    def forward(self, batch):
        x = batch["hidden_states"] if batch["hidden_states"] is not None else batch["input_ids"]
        x = self.embed_tokens(x)
        for layer in self.decoder_layers.values():
            x = layer(x)[0]
        x = self.norm(x)
        return self.lm_head(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad, retain_graph=False, create_graph=False)
        return input_tensor.grad if input_tensor is not None else None

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

class ShardedLlamaModel(ShardedModel):
    def _setup_model_components(self, model: AutoModelForCausalLM):
        self.embed_tokens = model.model.embed_tokens if self.world.is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.model.layers[i] for i in self.distribute_layers(model.config.num_hidden_layers)})
        self.norm = model.model.norm if self.world.is_last_stage else nn.Identity()
        self.lm_head = model.lm_head if self.world.is_last_stage else nn.Identity()

class ShardedGPTModel(ShardedModel):
    def _setup_model_components(self, model: AutoModelForCausalLM):
        self.embed_tokens = model.transformer.wte if self.world.is_first_stage else nn.Identity()
        self.decoder_layers = nn.ModuleDict({str(i): model.transformer.h[i] for i in self.distribute_layers(model.config.num_hidden_layers)})
        self.norm = model.transformer.ln_f if self.world.is_last_stage else nn.Identity()
        self.lm_head = model.lm_head if self.world.is_last_stage else nn.Identity()