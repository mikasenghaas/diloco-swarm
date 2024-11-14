from enum import Enum
from typing import Tuple, Optional

import torch
import torch.distributed as dist

from src.world import World

class Comm:
    def __init__(self, world: World, shape: Tuple[int, ...], dtype: torch.dtype):
        self.world = world
        self.shape = shape
        self.dtype = dtype

    def synchronize(self, req: Optional[dist.Work] = None) -> None:
        if req is not None: req.wait()
        torch.cuda.synchronize()

    def send_to(self, tensor: torch.Tensor, peer: int, tag: int = 0) -> None:
        req = dist.isend(tensor, dst=peer, tag=tag)
        self.synchronize(req)

    def recv_from(self, peer: int, tag: int = 0) -> torch.Tensor:
        tensor = torch.empty(self.shape, requires_grad=True, device="cuda", dtype=self.dtype)
        req = dist.irecv(tensor, src=peer, tag=tag)
        self.synchronize(req)
        return tensor

    def send_forward(self, tensor: torch.Tensor, tag: int = 0) -> None:
        if self.world.is_last_stage: return
        self.send_to(tensor, self.world.next_rank(), tag)

    def recv_forward(self, tag: int = 0) -> Optional[torch.Tensor]:
        if self.world.is_first_stage: return None
        return self.recv_from(self.world.prev_rank(), tag)

    def send_backward(self, tensor: torch.Tensor, tag: int = 0) -> None:
        if self.world.is_first_stage: return
        self.send_to(tensor, self.world.prev_rank(), tag)

    def recv_backward(self, tag: int = 0) -> Optional[torch.Tensor]:
        if self.world.is_last_stage: return None
        return self.recv_from(self.world.next_rank(), tag)

    def __str__(self):
        return f"Comm(world={self.world}, shape={self.shape}, dtype={self.dtype})"
