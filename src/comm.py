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

    def send_forward(self, tensor: torch.Tensor, tag: int) -> None:
        if self.world.is_last_stage: return
        req = dist.isend(tensor, dst=self.world.next_rank(), tag=tag)
        self.synchronize(req)

    def recv_forward(self, tag: int) -> Optional[torch.Tensor]:
        tensor = torch.empty(self.shape, requires_grad=True, device="cuda", dtype=self.dtype)
        if self.world.is_first_stage: return None
        req = dist.irecv(tensor, src=self.world.prev_rank(), tag=tag)
        self.synchronize(req)
        return tensor

    def send_backward(self, tensor: torch.Tensor, tag: int) -> None:
        if self.world.is_first_stage: return
        req = dist.isend(tensor, dst=self.world.prev_rank(), tag=tag)
        self.synchronize(req)

    def recv_backward(self, tag: int) -> Optional[torch.Tensor]:
        if self.world.is_last_stage: return None
        tensor = torch.empty(self.shape, requires_grad=True, device="cuda", dtype=self.dtype)
        req = dist.irecv(tensor, src=self.world.next_rank(), tag=tag)
        self.synchronize(req)
        return tensor

    def __str__(self):
        return f"Comm(world={self.world}, shape={self.shape}, dtype={self.dtype})"
