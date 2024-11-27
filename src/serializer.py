from typing import Tuple

import torch

Metadata = Tuple[int, int] # (root, local_micro_step)
SerializedType = torch.Tensor
DeserializedType = Tuple[torch.Tensor, Metadata]

class Serializer: # todo: can improve to by factor O(min(d1, ..., dn)) or using sparse tensors
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = (2, *shape)

    def serialize(self, tensor: torch.Tensor, metadata: Metadata) -> SerializedType:
        metadata_tensor = torch.empty(tensor.numel(), device=tensor.device)
        metadata_tensor[0], metadata_tensor[1] = float(metadata[0]), float(metadata[1])
        metadata_tensor = metadata_tensor.reshape(tensor.shape)
        return torch.cat([metadata_tensor.unsqueeze(0), tensor.unsqueeze(0)], dim=0)

    def deserialize(self, serialized: SerializedType) -> DeserializedType:
        metadata_tensor = serialized[0].flatten()
        tensor = serialized[1:]
        root = int(metadata_tensor[0].item())
        local_micro_step = int(metadata_tensor[1].item())
        return tensor.squeeze(0), (root, local_micro_step)