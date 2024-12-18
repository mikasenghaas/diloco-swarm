import time
import random
import threading
from queue import Queue
from typing import Tuple, List, Optional, Literal

import torch
import torch.nn as nn
import torch.distributed as dist

from src.world import World
from src.metrics import Outputs
from src.logger import Logger, Level
from src.serializer import Serializer, Metadata

class Comm:
    def __init__(self, world: World, shape: Tuple[int, ...], activations_shape: Tuple[int, ...], input_ids_shape: Tuple[int, ...], dtype: torch.dtype, model: nn.Module, serializer: Serializer, device: torch.device, logger: Logger, timeout: float):
        self.world, self.dtype, self.model, self.serializer, self.device, self.timeout, self.logger = world, dtype, model, serializer, device, timeout, logger
        self.shape, self.activations_shape, self.input_ids_shape = shape, activations_shape, input_ids_shape

        self.forward_send_queue, self.backward_recv_queue = Queue(), Queue()
        self.forward_recv_queue, self.backward_send_queue = Queue(), Queue()
        if self.world.has_next_stage:
            self.logger.log_message(f"Creating threads for sending/ receiving training activations of shape {self.activations_shape} and dtype {self.dtype}", Level.DEBUG, master_only=False)
            threading.Thread(target=self._send_loop, args=(self.forward_send_queue, self.world.next_stage_group), daemon=True).start()
            threading.Thread(target=self._recv_loop, args=(0, self.backward_recv_queue, self.shape, self.dtype, self.world.next_stage_group), daemon=True).start()

        if self.world.has_prev_stage:
            self.logger.log_message(f"Creating threads for sending/ receiving training activations of shape {self.shape} and dtype {self.dtype}", Level.DEBUG, master_only=False)
            threading.Thread(target=self._send_loop, args=(self.backward_send_queue, self.world.prev_stage_group), daemon=True).start()
            threading.Thread(target=self._recv_loop, args=(0, self.forward_recv_queue, self.shape, self.dtype, self.world.prev_stage_group), daemon=True).start()

        if self.activations_shape:
            if not self.world.is_last_stage:
                self.logger.log_message(f"Creating thread for sending inference activations of shape {self.activations_shape} and dtype {self.dtype}", Level.DEBUG, master_only=False)
                self.activations_send_queue = Queue()
                threading.Thread(target=self._send_loop, args=(self.activations_send_queue, self.world.next_stage_group, False), daemon=True).start()
            if not self.world.is_first_stage:
                self.logger.log_message(f"Creating thread for receiving inference activations of shape {self.activations_shape} and dtype {self.dtype}", Level.DEBUG, master_only=False)
                self.activations_recv_queue = Queue()
                threading.Thread(target=self._recv_loop, args=(1, self.activations_recv_queue, self.activations_shape, self.dtype, self.world.prev_stage_group, False, False), daemon=True).start()

        if self.input_ids_shape:
            if self.world.is_last_stage:
                self.input_ids_send_queue = Queue()
                self.logger.log_message(f"Creating thread for sending input_ids of shape {self.input_ids_shape} and dtype {torch.long}", Level.DEBUG, master_only=False)
                threading.Thread(target=self._send_loop, args=(self.input_ids_send_queue, self.world.first_last_stage_group, False), daemon=True).start()
            if self.world.is_first_stage:
                self.input_ids_recv_queue = Queue()
                self.logger.log_message(f"Creating thread for receiving input_ids of shape {self.input_ids_shape} and dtype {torch.long}", Level.DEBUG, master_only=False)
                threading.Thread(target=self._recv_loop, args=(1, self.input_ids_recv_queue, self.input_ids_shape, torch.long, self.world.first_last_stage_group, False, False), daemon=True).start()

        dist.barrier()

    def send_forward(self, tensor: torch.Tensor, metadata: Metadata) -> None:
        if not self.world.has_next_stage: return
        dst = random.choice(self.world.stage2ranks[self.world.stage + 1])
        self.forward_send_queue.put((dst, 0, tensor, metadata))
        self.logger.log_message(f"Sent activations (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, root={metadata[0]}, local_micro_step={metadata[1]}) to rank {dst}", Level.DEBUG, master_only=False)

    def send_backward(self, dst: int, tensor: torch.Tensor, metadata: Metadata) -> None:
        if not self.world.has_prev_stage: return
        self.backward_send_queue.put((dst, 0, tensor, metadata))
        self.logger.log_message(f"Sent gradients (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, root={metadata[0]}, local_micro_step={metadata[1]}) to rank {dst}", Level.DEBUG, master_only=False)

    def recv_forward(self, device: torch.device) -> Optional[Tuple[int, torch.Tensor, Metadata]]:
        src, tensor, metadata = self.forward_recv_queue.get()
        self.logger.log_message(f"Received activations (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, root={metadata[0]}, local_micro_step={metadata[1]}) from rank {src}", Level.DEBUG, master_only=False)
        return src, tensor.to(device), metadata

    def recv_backward(self, device: torch.device) -> Optional[Tuple[int, torch.Tensor, Metadata]]:
        src, tensor, metadata = self.backward_recv_queue.get()
        self.logger.log_message(f"Received gradients (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, root={metadata[0]}, local_micro_step={metadata[1]}) from rank {src}", Level.DEBUG, master_only=False)
        return src, tensor.to(device), metadata

    def send_activations(self, activations: torch.Tensor) -> None:
        if not self.world.has_next_stage: return
        self.activations_send_queue.put((self.world.next_stage_leader, 1, activations))
        self.logger.log_message(f"Sent inference activations (shape={activations.shape}, dtype={activations.dtype}, device={activations.device}) to rank {self.world.next_stage_leader}", Level.DEBUG, master_only=False)

    def recv_activations(self, device: torch.device) -> Optional[torch.Tensor]:
        if not self.world.has_prev_stage: return None
        while True:
            if self.activations_recv_queue.empty(): time.sleep(self.timeout); continue
            src, activations = self.activations_recv_queue.get()
            self.logger.log_message(f"Received inference activations (shape={activations.shape}, dtype={activations.dtype}, device={activations.device}) from rank {src}", Level.DEBUG, master_only=False)
            return activations.to(device)

    def send_input_ids(self, input_ids: torch.Tensor) -> None:
        if not (self.world.is_last_stage and self.world.is_leader) or self.world.is_first_stage: return
        assert input_ids.shape == self.input_ids_shape
        self.input_ids_send_queue.put((self.world.first_stage_leader,1, input_ids))
        self.logger.log_message(f"Sent input_ids (shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}) to rank {self.world.first_stage_leader}", Level.DEBUG, master_only=False)

    def recv_input_ids(self, device: torch.device) -> Optional[torch.Tensor]:
        if not (self.world.is_first_stage and self.world.is_leader): return None
        while True:
            if self.input_ids_recv_queue.empty(): time.sleep(self.timeout); continue
            src, input_ids = self.input_ids_recv_queue.get()
            self.logger.log_message(f"Received input_ids (shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}) from rank {src}", Level.DEBUG, master_only=False)
            return input_ids.to(device)

    def load_input_ids_queue(self, input_ids: torch.Tensor) -> None:
        if not (self.world.is_first_stage and self.world.is_leader): return
        self.input_ids_recv_queue.put((-1, input_ids))

    def load_forward_queue(self, local_micro_step: int, input_tensor: torch.Tensor) -> None:
        self.forward_recv_queue.put((-1, input_tensor, (self.world.rank, local_micro_step)))

    def can_receive_forward(self) -> bool:
        return not self.forward_recv_queue.empty()

    def can_receive_backward(self) -> bool:
        return not self.backward_recv_queue.empty()
    
    def can_receive(self) -> bool:
        return self.can_receive_forward() or self.can_receive_backward()

    def sync_gradients(self) -> None:
        if len(self.world.stage2ranks[self.world.stage]) == 1: return
        for param in self.model.parameters():
            if param.grad is None: param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.world.curr_stage_group)
    
    def sync_outputs(self, outputs: Outputs) -> None:
        peers = self.world.stage2ranks[self.world.stage]
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

    def _recv_loop(self, type: Literal[0, 1], recv_queue: Queue, shape: Tuple[int, ...], dtype: torch.dtype, group: Optional[dist.ProcessGroup] = None, requires_grad: bool = True, serialize: bool = True):
        while True:
            tensor = torch.empty(shape, device="cpu", dtype=dtype, requires_grad=requires_grad)
            src = dist.recv(tensor, group=group, tag=type)
            # self.logger.log_message(f"Received tensor (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}) from rank {src}", Level.DEBUG, master_only=False)
            if serialize:
                tensor, metadata = self.serializer.deserialize(tensor)
                recv_queue.put((src, tensor, metadata))
            else:
                recv_queue.put((src, tensor))

    def _send_loop(self, send_queue: Queue, group: Optional[dist.ProcessGroup] = None, serialize: bool = True):
        while True:
            if send_queue.empty(): time.sleep(self.timeout); continue
            item = send_queue.get()
            if serialize:
                (dst, type, tensor, metadata) = item
                serialized = self.serializer.serialize(tensor, metadata)
                # self.logger.log_message(f"Sending serialized {serialized.shape} to {dst}", Level.DEBUG, master_only=False)
                dist.send(serialized.to("cpu"), dst=dst, group=group, tag=type)
            else:
                (dst, type, tensor) = item
                # self.logger.log_message(f"Sending unserialized {tensor.shape} to {dst}", Level.DEBUG, master_only=False)
                dist.send(tensor.to("cpu"), dst=dst, group=group, tag=type)
