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
    def __init__(self, world: World, shape: Tuple[int, ...], activations_shape: Tuple[int, ...], input_ids_shape: Tuple[int, ...], dtype: torch.dtype, model: nn.Module, serializer: Serializer, device: torch.device, logger: Logger, timeout: float):
        self.world, self.dtype, self.model, self.serializer, self.device, self.timeout, self.logger = world, dtype, model, serializer, device, timeout, logger
        self.shape, self.activations_shape, self.input_ids_shape = shape, activations_shape, input_ids_shape
        self.forward_send_queue, self.backward_send_queue = Queue(), Queue()
        self.forward_recv_queue, self.backward_recv_queue = Queue(), Queue()

        self.logger.log_message(f"Prev stage group: {self.world.prev_stage_group}")
        self.logger.log_message(f"Next stage group: {self.world.next_stage_group}")
        self.logger.log_message(f"First last stage group: {self.world.first_last_stage_group}")

        # threading.Thread(target=self._send_loop, args=(self.forward_send_queue, self.world.next_stage_group), daemon=True).start()
        # threading.Thread(target=self._send_loop, args=(self.backward_send_queue, self.world.prev_stage_group), daemon=True).start()
        # threading.Thread(target=self._recv_loop, args=(self.forward_recv_queue, self.shape, self.dtype, self.world.prev_stage_group), daemon=True).start()
        # threading.Thread(target=self._recv_loop, args=(self.backward_recv_queue, self.shape, self.dtype, self.world.next_stage_group), daemon=True).start()
        if self.activations_shape:
            self.activations_send_queue, self.activations_recv_queue = Queue(), Queue()
            self.logger.log_message(f"Creating threads for sending/ receiving activations of shape {self.activations_shape} and dtype {self.dtype}")
            if not self.world.is_last_stage:
                threading.Thread(target=self._send_loop, args=(self.activations_send_queue, self.world.next_stage_group, False), daemon=True).start()
            if not self.world.is_first_stage:
                threading.Thread(target=self._recv_loop, args=(self.activations_recv_queue, self.activations_shape, self.dtype, self.world.prev_stage_group, False, False), daemon=True).start()
        if self.input_ids_shape:
            self.input_ids_send_queue, self.input_ids_recv_queue = Queue(), Queue()
            self.logger.log_message(f"Creating threads for sending/ receiving input_ids of shape {self.input_ids_shape} and dtype {torch.long}")
            if self.world.is_last_stage:
                threading.Thread(target=self._send_loop, args=(self.input_ids_send_queue, self.world.first_last_stage_group, False), daemon=True).start()
            if self.world.is_first_stage:
                threading.Thread(target=self._recv_loop, args=(self.input_ids_recv_queue, self.input_ids_shape, torch.long, self.world.first_last_stage_group, False, False), daemon=True).start()

        dist.barrier()

    def send_forward(self, tensor: torch.Tensor, metadata: Metadata) -> None:
        if self.world.is_last_stage: return
        dst = random.choice(self.world.next_stage_ranks)
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

    def send_activations(self, activations: torch.Tensor) -> None:
        if self.world.is_last_stage: return
        self.activations_send_queue.put((self.world.next_stage_leader, activations))
        self.logger.log_message(f"Sent activations (shape={activations.shape}, dtype={activations.dtype}, device={activations.device}) to rank {self.world.next_stage_leader}", Level.DEBUG)

    def recv_activations(self) -> Optional[torch.Tensor]:
        if self.world.is_first_stage: return None
        while True:
            if self.activations_recv_queue.empty(): time.sleep(self.timeout); continue
            src, activations = self.activations_recv_queue.get()
            self.logger.log_message(f"Received activations (shape={activations.shape}, dtype={activations.dtype}, device={activations.device}) from rank {src}", Level.DEBUG)
            return activations

    def send_input_ids(self, input_ids: torch.Tensor) -> None:
        if not (self.world.is_last_stage and self.world.is_leader): return
        assert input_ids.shape == self.input_ids_shape
        self.input_ids_send_queue.put((self.world.first_stage_leader, input_ids))
        self.logger.log_message(f"Sent input_ids (shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}) to rank {self.world.first_stage_leader}", Level.DEBUG)

    def recv_input_ids(self) -> Optional[torch.Tensor]:
        if not (self.world.is_first_stage and self.world.is_leader): return None
        while True:
            if self.input_ids_recv_queue.empty(): time.sleep(self.timeout); continue
            src, input_ids = self.input_ids_recv_queue.get()
            self.logger.log_message(f"Received input_ids (shape={input_ids.shape}, dtype={input_ids.dtype}, device={input_ids.device}) from rank {src}", Level.DEBUG)
            return input_ids

    def load_input_ids_queue(self, input_ids: torch.Tensor) -> None:
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

    def _recv_loop(self, recv_queue: Queue, shape: Tuple[int, ...], dtype: torch.dtype, group: Optional[dist.ProcessGroup] = None, requires_grad: bool = True, serialize: bool = True):
        while True:
            tensor = torch.empty(shape, device="cpu", dtype=dtype, requires_grad=requires_grad)
            src = dist.recv(tensor, group=group)
            self.logger.log_message(f"Received something of shape {tensor.shape} from {src}", Level.DEBUG)
            if serialize:
                tensor, metadata = self.serializer.deserialize(tensor)
                recv_queue.put((src, tensor.to(self.device), metadata))
            else:
                recv_queue.put((src, tensor.to(self.device)))

    def _send_loop(self, send_queue: Queue, group: Optional[dist.ProcessGroup] = None, serialize: bool = True):
        while True:
            if send_queue.empty(): time.sleep(self.timeout); continue
            item = send_queue.get()
            if serialize:
                (dst, tensor, metadata) = item
                serialized = self.serializer.serialize(tensor, metadata)
                self.logger.log_message(f"Sending serialized {serialized.shape} to {dst}", Level.DEBUG)
                dist.send(serialized.to("cpu"), dst=dst, group=group)
            else:
                (dst, tensor) = item
                self.logger.log_message(f"Sending something unserialized {tensor.shape} to {dst} (group={group})", Level.DEBUG)
                dist.send(tensor.to("cpu"), dst=dst, group=group)
                self.logger.log_message(f"Sent something unserialized {tensor.shape} to {dst}", Level.DEBUG)