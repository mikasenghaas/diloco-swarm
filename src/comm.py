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
from src.serializer import Serializer, Metadata

class SendThread:
    """
    Implements a sending thread that sends tensor of a specified shape and
    tag to a destination process group by enqueing them into a queue. The thread
    is started by default. Tensor may be serialized to include metadata before
    sending.
    """
    TIMEOUT = 1e-4
    def __init__(self, shape: Tuple[int, ...], group: dist.ProcessGroup, tag: int = 0, serialize: bool = True, start: bool = True, **kwargs):
        self.shape, self.group, self.tag, self.serialize = shape, group, tag, serialize
        self.logger, self.queue = kwargs.get("logger"), Queue()
        if serialize: self.serializer = Serializer(shape); self.shape = self.serializer.shape
        if start: threading.Thread(target=self._send_loop, daemon=True).start()

    def send(self, dst: int, tensor: torch.Tensor, metadata: Optional[Metadata]) -> None:
        return self.queue.put((dst, tensor, metadata))

    def _send_loop(self):
        while True:
            if self.queue.empty(): time.sleep(self.TIMEOUT); continue
            dst, tensor, metadata = self.queue.get()
            if self.logger: self.logger.log_message(f"Sent tensor to {dst} (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, metadata={metadata})", level=Level.DEBUG, master=False)
            if self.serialize: tensor = self.serializer.serialize(tensor, metadata)
            dist.send(tensor.to("cpu"), dst=dst, group=self.group, tag=self.tag)

class RecvThread:
    """
    Implements a receiving thread that receives tensor of a specified shape and
    tag from a source process group by enqueing them into a queue. The thread
    is started by default. Tensor may be deserialized to extract metadata after
    receiving.
    """
    def __init__(self, shape: Tuple[int, ...], group: dist.ProcessGroup, tag: int = 0, requires_grad: bool = True, dtype: torch.dtype = torch.float32, serialize: bool = True, start: bool = True, **kwargs):
        self.shape, self.group, self.tag, self.requires_grad, self.dtype, self.serialize = shape, group, tag, requires_grad, dtype, serialize
        self.logger, self.queue = kwargs.get("logger"), Queue()
        if serialize: self.serializer = Serializer(shape); self.shape = self.serializer.shape
        if start: threading.Thread(target=self._recv_loop, daemon=True).start()

    @property
    def can_receive(self) -> bool:
        return not self.queue.empty()

    def load(self, tensor: Optional[torch.Tensor], metadata: Optional[Metadata]) -> None:
        return self.queue.put((-1, tensor, metadata))

    def receive(self) -> Tuple[int, torch.Tensor, Optional[Metadata]]:
        return self.queue.get()

    def _recv_loop(self):
        while True:
            tensor, metadata = torch.empty(self.shape, dtype=self.dtype, requires_grad=self.requires_grad), None
            src = dist.recv(tensor, group=self.group, tag=self.tag)
            if self.serialize: tensor, metadata = self.serializer.deserialize(tensor)
            if self.logger: self.logger.log_message(f"Received tensor from {src} (shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, metadata={metadata})", level=Level.DEBUG, master=False)
            self.queue.put((src, tensor, metadata))

class TrainingComm():
    """
    Implements communication for training. A worker has a sender and receiver thread for each adjacent stage, i.e.
    - First stage has a sender and receiver for the next stage
    - Last stage has a sender and receiver for the previous stage
    - Middle stages have senders and receivers for both the next and previous stages
    All threads communicate tensor of shape (2, batch_size, seq_len, hidden_size) where the first dimension is 
    added to store metadata which is serialized into the tensor before sending and deserialized after receiving.
    The class also implements gradient synchronization and output aggregation that can be called after a step.
    """
    def __init__(self, world: World, shape: Tuple[int, ...], logger: Logger):
        self.world, self.shape, self.logger = world, shape, logger
        training_kwargs = {"tag": 0, "serialize": True, "requires_grad": True, "logger": logger}
        self.forward_send_thread = SendThread(shape, group=self.world.next_stage_group, start=self.world.has_next_stage, **training_kwargs)
        self.backward_recv_thread = RecvThread(shape, group=self.world.next_stage_group, start=self.world.has_next_stage, **training_kwargs)
        self.backward_send_thread = SendThread(shape, group=self.world.prev_stage_group, start=self.world.has_prev_stage, **training_kwargs)
        self.forward_recv_thread = RecvThread(shape, group=self.world.prev_stage_group, start=self.world.has_prev_stage, **training_kwargs)

    def send_forward(self, tensor: torch.Tensor, metadata: Metadata) -> None:
        if not self.world.has_next_stage: return
        dst = random.choice(self.world.stage2ranks[self.world.stage + 1]) # Random next stage rank
        self.forward_send_thread.send(dst=dst, tensor=tensor, metadata=metadata)

    def send_backward(self, dst: int, tensor: torch.Tensor, metadata: Metadata) -> None:
        if not self.world.has_prev_stage: return
        self.backward_send_thread.send(dst=dst, tensor=tensor, metadata=metadata)

    def recv_forward(self) -> Tuple[int, torch.Tensor, Optional[Metadata]]:
        return self.forward_recv_thread.receive()

    def recv_backward(self) -> Tuple[int, torch.Tensor, Optional[Metadata]]:
        return self.backward_recv_thread.receive()

    def load_forward(self, metadata: Metadata) -> None:
        self.forward_recv_thread.load(tensor=None, metadata=metadata)

    def sync_gradients(self, model: nn.Module) -> None:
        num_peers = len(self.world.stage2ranks[self.world.stage])
        if num_peers == 1: return
        for param in model.parameters():
            if param.grad is None: param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.world.curr_stage_group)
            param.grad /= num_peers

    def sync_outputs(self, outputs: Outputs) -> None:
        peers = self.world.stage2ranks[self.world.stage]
        if len(peers) == 1: return outputs
        all_outputs : List[Outputs] = [None] * len(peers)
        dist.all_gather_object(all_outputs, outputs, group=self.world.curr_stage_group)

        return Outputs(
            step=outputs.step,
            time=outputs.time,
            loss=sum([output.loss for output in all_outputs]),
            tokens=sum([output.tokens for output in all_outputs]),
            num_micro_batches=sum([output.num_micro_batches for output in all_outputs]),
            lr=outputs.lr,
            norm=outputs.norm,
            micro_step_time=outputs.micro_step_time
        )

class InferenceComm():
    """
    Implements communication for inference. The leaders of each stage form a
    circular ring for communication. All but the last to first stage communicate
    activations float32 tensors of shape (sample_size, seq_len, hidden_size) and
    the last stage communicates input ids int64 tensors of shape (sample_size,
    seq_len). Tensor are not serialized.
    """
    def __init__(self, world: World, shape: Tuple[int, ...], logger: Logger):
        self.world, self.shape, self.logger = world, shape, logger
        inference_kwargs = {"tag": 1, "serialize": False, "requires_grad": False, "start": True, "logger": logger}
        if not self.world.is_leader: return
        if self.world.is_first_stage:
            self.send_thread = SendThread(self.shape, group=self.world.next_stage_group, **inference_kwargs)
            self.recv_thread = RecvThread(self.shape[:-1], group=self.world.first_last_stage_group, dtype=torch.long, **inference_kwargs)
        elif self.world.is_last_stage:
            self.send_thread = SendThread(self.shape[:-1], group=self.world.first_last_stage_group, **inference_kwargs)
            self.recv_thread = RecvThread(self.shape, group=self.world.prev_stage_group, **inference_kwargs)
        else:
            self.send_thread = SendThread(self.shape, group=self.world.next_stage_group, **inference_kwargs)
            self.recv_thread = RecvThread(self.shape, group=self.world.prev_stage_group, **inference_kwargs)
        self.receive_tensor_type = "hidden_states" if not self.world.is_first_stage else "input_ids"
        self.dst = self.world.stage2leader[self.world.stage+1] if not self.world.is_last_stage else self.world.stage2leader[0]

    def receive(self) -> Tuple[str, torch.Tensor]:
        return self.receive_tensor_type, self.recv_thread.receive()[1]

    def send(self, tensor: torch.Tensor) -> None:
        if self.world.rank == self.dst: self.load(tensor); return
        self.send_thread.send(dst=self.dst, tensor=tensor, metadata=None)
    
    def load(self, tensor: torch.Tensor) -> None:
        self.recv_thread.load(tensor=tensor, metadata=None)