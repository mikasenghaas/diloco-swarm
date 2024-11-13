from transformers import PreTrainedModel, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

class HFConfig(PretrainedConfig):
    model_type = "custom_causal_lm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = kwargs.get("name")

class HFModel(PreTrainedModel):
    def __init__(self, config, base_model):
        super().__init__(config)
        self.config = config
        self.model = base_model.transformer
        self.lm_head = base_model.lm_head