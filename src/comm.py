import time
import random
import threading
from queue import Queue
from typing import Tuple, List, Optional


import torch
import torch.nn as nn
import torch.distributed as dist

from src.world import World, SwarmWorld
from src.logger import CustomLogger, Level
from src.metrics import Outputs
from src.serializer import SwarmSerializer, Metadata, DeserializedType

TIMEOUT = 0.001

class Comm:
    def __init__(self, world: World, shape: Tuple[int, ...], dtype: torch.dtype):
        self.world = world
        self.shape = shape
        self.dtype = dtype

    def synchronize(self, req: Optional[dist.Work] = None) -> None:
        if req is not None: req.wait()
        torch.cuda.synchronize()

    def send_to(self, tensor: torch.Tensor, peer: int, tag: int = 0) -> None:
        req = dist.isend(tensor.detach().clone(), dst=peer, tag=tag)
        self.synchronize(req)

    def recv_from(self, peer: int, tag: int = 0, device: Optional[torch.device] = None, requires_grad: bool = True) -> torch.Tensor:
        tensor = torch.empty(self.shape, requires_grad=requires_grad, device=device, dtype=self.dtype)
        req = dist.irecv(tensor, src=peer, tag=tag)
        self.synchronize(req)
        return tensor

    def send_forward(self, tensor: torch.Tensor, tag: int = 0) -> None:
        if self.world.is_last_stage: return
        self.send_to(tensor, self.world.next_rank(), tag)

    def recv_forward(self, tag: int = 0, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.world.is_first_stage: return None
        return self.recv_from(self.world.prev_rank(), tag, device)

    def send_backward(self, tensor: torch.Tensor, tag: int = 0) -> None:
        if self.world.is_first_stage: return
        self.send_to(tensor, self.world.prev_rank(), tag)

    def recv_backward(self, tag: int = 0, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if self.world.is_last_stage: return None
        return self.recv_from(self.world.next_rank(), tag, device)

    def __str__(self):
        return f"Comm(world={self.world}, shape={self.shape}, dtype={self.dtype})"

class SwarmComm:
    def __init__(self, model: nn.Module, world: SwarmWorld, serializer: SwarmSerializer, shape: Tuple, device: torch.device, logger: CustomLogger):
        self.model, self.world, self.serializer, self.shape, self.device, self.logger = model, world, serializer, shape, device, logger
        self.forward_send_queue, self.backward_send_queue = Queue(), Queue()
        self.forward_recv_queue, self.backward_recv_queue = Queue(), Queue()
        threading.Thread(target=self._receive_loop, args=(self.forward_recv_queue, self.shape, self.world.prev_stage_group), daemon=True).start()
        threading.Thread(target=self._receive_loop, args=(self.backward_recv_queue, self.shape, self.world.next_stage_group), daemon=True).start()
        threading.Thread(target=self._send_loop, args=(self.forward_send_queue, self.world.next_stage_group), daemon=True).start()
        threading.Thread(target=self._send_loop, args=(self.backward_send_queue, self.world.prev_stage_group), daemon=True).start()

    def send_forward(self, tensor: torch.Tensor, metadata: Metadata) -> None:
        if self.world.is_last_stage: return
        dst = random.choice(self.world.get_stage_ranks(self.world.stage + 1))
        self.forward_send_queue.put((dst, tensor, metadata))
        self.logger.log_message(f"Sent activations to {dst} {metadata}", Level.DEBUG)

    def send_backward(self, dst: int, tensor: torch.Tensor, metadata: Metadata) -> None:
        if self.world.is_first_stage: return
        self.backward_send_queue.put((dst, tensor, metadata))
        self.logger.log_message(f"Sent gradients to {dst} {metadata}", Level.DEBUG)

    def recv_forward(self) -> Optional[Tuple[int, DeserializedType]]:
        src, tensor, metadata = self.forward_recv_queue.get()
        self.logger.log_message(f"Received activations from {src} {metadata}", Level.DEBUG)
        return src, tensor, metadata

    def recv_backward(self) -> Optional[Tuple[int, DeserializedType]]:
        if self.world.is_last_stage: return None
        src, tensor, metadata = self.backward_recv_queue.get()
        self.logger.log_message(f"Received gradients from {src} {metadata}", Level.DEBUG)
        return src, tensor, metadata

    def load_forward_queue(self, local_micro_step: int, input_tensor: torch.Tensor) -> None:
        self.forward_recv_queue.put((-1, input_tensor, (self.world.rank, local_micro_step)))

    def can_receive_forward(self) -> bool:
        return not self.forward_recv_queue.empty()

    def can_receive_backward(self) -> bool:
        return not self.backward_recv_queue.empty()

    def can_receive(self) -> bool:
        return self.can_receive_forward() or self.can_receive_backward()

    def sync_gradients(self) -> None:
        for param in self.model.parameters():
            if param.grad is None: param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.world.curr_stage_group)
    
    def sync_outputs(self, outputs: Outputs) -> None:
        all_outputs : List[Outputs] = [None] * len(self.world.get_stage_ranks(self.world.stage))
        dist.all_gather_object(all_outputs, outputs, group=self.world.curr_stage_group)

        return Outputs(
            step=outputs.step,
            time=outputs.time, # compute avg?
            loss=sum([output.loss for output in all_outputs]),
            num_tokens=sum([output.num_tokens for output in all_outputs]),
            num_examples=sum([output.num_examples for output in all_outputs]),
            lr=outputs.lr,
            norm=outputs.norm,
            micro_step_time=outputs.micro_step_time
        )

    def _receive_loop(self, recv_queue: Queue, shape: Tuple[int, int], group: dist.ProcessGroup | None = None):
        while True:
            serialized = torch.empty(shape, device="cpu", requires_grad=True) # gloo backend
            src = dist.recv(serialized, group=group)
            tensor, metadata = self.serializer.deserialize(serialized)
            recv_queue.put((src, tensor.to(self.device), metadata))

    def _send_loop(self, send_queue: Queue, group: dist.ProcessGroup | None = None):
        while True:
            if send_queue.empty():
                time.sleep(TIMEOUT)
                continue
            dst, tensor, metadata = send_queue.get()
            serialized = self.serializer.serialize(tensor, metadata)
            dist.send(serialized.to("cpu"), dst=dst, group=group) # gloo backend
