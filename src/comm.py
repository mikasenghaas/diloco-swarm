import time
import random
import threading
from queue import Queue
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.distributed as dist

from src.world import World
from src.metrics import Outputs
from src.logger import Logger, Level
from src.serializer import Serializer, Metadata, DeserializedType

class Comm:
    def __init__(self, world: World, shape: Tuple[int, ...], dtype: torch.dtype, logger: Logger):
        self.world, self.shape, self.dtype, self.logger = world, shape, dtype, logger

    def send_to(self, tensor: torch.Tensor, dst: int) -> None:
        dist.send(tensor.detach().clone(), dst=dst)

    def recv_from(self, src: int, device: Optional[torch.device] = None, requires_grad: bool = False) -> torch.Tensor:
        tensor = torch.empty(self.shape, dtype=self.dtype, requires_grad=requires_grad, device=device)
        dist.recv(tensor, src=src)
        return tensor

    def recv_forward(self, device: Optional[torch.device] = None, requires_grad: bool = False) -> torch.Tensor:
        if self.world.is_first_stage: return None
        src = self.world.get_stage_ranks(self.world.stage - 1)[0]
        self.logger.log_message(f"Receiving activations from rank {src}", Level.DEBUG)
        return self.recv_from(src, device, requires_grad)

    def send_forward(self, tensor: torch.Tensor) -> None:
        if self.world.is_last_stage: return
        dst = self.world.get_stage_ranks(self.world.stage + 1)[0]
        self.logger.log_message(f"Sending activations to rank {dst}", Level.DEBUG)
        self.send_to(tensor, dst)

    def __repr__(self):
        return f"Comm(world={self.world}, shape={self.shape}, dtype={self.dtype})"

class SwarmComm(Comm):
    def __init__(self, world: World, shape: Tuple[int, ...], dtype: torch.dtype, model: nn.Module, serializer: Serializer, device: torch.device, logger: Logger, timeout: float):
        super().__init__(world, shape, dtype, logger)
        self.model, self.serializer, self.device, self.timeout = model, serializer, device, timeout
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
        self.logger.log_message(f"Sent activations (root={metadata[0]}, local_micro_step={metadata[1]}) to rank {dst} {metadata}", Level.DEBUG)

    def send_backward(self, dst: int, tensor: torch.Tensor, metadata: Metadata) -> None:
        if self.world.is_first_stage: return
        self.backward_send_queue.put((dst, tensor, metadata))
        self.logger.log_message(f"Sent gradients (root={metadata[0]}, local_micro_step={metadata[1]}) to rank {dst} {metadata}", Level.DEBUG)

    def recv_forward(self) -> Optional[Tuple[int, DeserializedType]]:
        src, tensor, metadata = self.forward_recv_queue.get()
        self.logger.log_message(f"Received activations (root={metadata[0]}, local_micro_step={metadata[1]}) from rank {src} {metadata}", Level.DEBUG)
        return src, tensor, metadata

    def recv_backward(self) -> Optional[Tuple[int, DeserializedType]]:
        if self.world.is_last_stage: return None
        src, tensor, metadata = self.backward_recv_queue.get()
        self.logger.log_message(f"Received gradients (root={metadata[0]}, local_micro_step={metadata[1]}) from rank {src} {metadata}", Level.DEBUG)
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
        peers = self.world.get_stage_ranks(self.world.stage)
        if len(peers) == 1: return
        for param in self.model.parameters():
            if param.grad is None: param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.world.curr_stage_group)
    
    def sync_outputs(self, outputs: Outputs) -> None:
        peers = self.world.get_stage_ranks(self.world.stage)
        if len(peers) == 1: return outputs
        all_outputs : List[Outputs] = [None] * len(peers)
        dist.all_gather_object(all_outputs, outputs, group=self.world.curr_stage_group)

        return Outputs(
            step=outputs.step,
            time=outputs.time, # compute avg?
            loss=sum([output.loss for output in all_outputs]),
            tokens=sum([output.tokens for output in all_outputs]),
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
                time.sleep(self.timeout); continue
            dst, tensor, metadata = send_queue.get()
            serialized = self.serializer.serialize(tensor, metadata)
            dist.send(serialized.to("cpu"), dst=dst, group=group) # gloo backend
