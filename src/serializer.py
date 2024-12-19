from typing import Tuple

import torch

Metadata = Tuple[int, int, int] # (train_step, root, local_micro_step)

class Serializer:
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = (2, *shape)

    def serialize(self, tensor: torch.Tensor, metadata: Metadata) -> torch.Tensor:
        metadata_tensor = torch.empty(tensor.numel(), device=tensor.device)
        metadata_tensor[0], metadata_tensor[1], metadata_tensor[2] = float(metadata[0]), float(metadata[1]), float(metadata[2])
        metadata_tensor = metadata_tensor.reshape(tensor.shape)
        return torch.cat([metadata_tensor.unsqueeze(0), tensor.unsqueeze(0)], dim=0)

    def deserialize(self, serialized: torch.Tensor) -> Tuple[torch.Tensor, Metadata]:
        metadata_tensor = serialized[0].flatten()
        tensor = serialized[1:]
        train_step = int(metadata_tensor[0].item())
        root = int(metadata_tensor[1].item())
        local_micro_step = int(metadata_tensor[2].item())
        return tensor.squeeze(0), (train_step, root, local_micro_step)
