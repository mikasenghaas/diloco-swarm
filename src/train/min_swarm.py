"""
Minimal SWARM implementation. Train a MLP to fit a quadratic function.

Runs on CPU with only dependency being PyTorch with the following command:

```bash
python min_swarm.py --world-size 2 --num-stages 2
```
"""
import time
import random
import threading
from queue import Queue
from typing import Dict, List, Tuple, Optional
from itertools import cycle

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler

TIMEOUT = 0.001 # critical to tune

class Data(Dataset): # quadratic function
    def __init__(self, length: int):
        self.x = torch.linspace(-5, 5, length).float().reshape(-1, 1).requires_grad_(True)
        self.y = self.x ** 2

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"inputs": self.x[idx], "targets": self.y[idx]}

class BatchData(Dataset):
    def __init__(self, batch: Dict[str, torch.Tensor]):
        self.batch = batch

    def __len__(self) -> int:
        return len(self.batch["inputs"])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] for k, v in self.batch.items()}

class DistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, rank: int, ranks: List[int], micro_batch_size: int):
        self.dataset, self.rank, self.ranks, self.micro_batch_size = dataset, rank, ranks, micro_batch_size
        self.num_ranks, self.num_samples = len(ranks), len(dataset)
        self.idx2rank = {i: rank for i, rank in enumerate(ranks)}
        self.rank2idx = {rank: i for i, rank in enumerate(ranks)}

    def __len__(self):
        samples_per_full_round = self.num_ranks * self.micro_batch_size
        
        full_rounds = self.num_samples // samples_per_full_round
        samples_from_full_rounds = full_rounds * self.micro_batch_size
        
        remaining_samples = self.num_samples % samples_per_full_round
        rank_start_in_last_round = self.rank2idx[self.rank] * self.micro_batch_size
        remaining_for_rank = max(0, min(self.micro_batch_size, remaining_samples - rank_start_in_last_round))
        
        return samples_from_full_rounds + remaining_for_rank

    def __iter__(self):
        for batch_start in range(0, len(self.dataset), self.num_ranks * self.micro_batch_size):
            rank_start = batch_start + (self.rank2idx[self.rank] * self.micro_batch_size)
            for idx in range(rank_start, min(rank_start + self.micro_batch_size, len(self.dataset))):
                yield idx

class Model(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([Model.block(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, 1)
    
    @staticmethod
    def block(hidden_dim: int) -> nn.Sequential:
        return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))

class ShardedModel(nn.Module):
    def __init__(self, model: nn.Module, stage: int, num_stages: int):
        super().__init__()
        self.stage, self.num_stages = stage, num_stages
        self.embed = model.embed if stage == 0 else nn.Identity()
        self.blocks = nn.ModuleList([model.blocks[i] for i in self.distribute_layers(len(model.blocks))])
        self.head = model.head if stage == num_stages - 1 else nn.Identity()
        del model

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // self.num_stages + (1 if i < num_layers % self.num_stages else 0) for i in range(self.num_stages)]
        start_layer = sum(layers_per_gpu[:self.stage])
        return list(range(start_layer, start_layer + layers_per_gpu[self.stage]))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed(inputs)
        for layer in self.blocks:
            x = layer(x)
        return self.head(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        return input_tensor.grad if input_tensor is not None else None

class SwarmWorld:
    def __init__(self, rank: int, world_size: int, num_stages: int, master_host: str, master_port: int):
        assert world_size >= num_stages, "World size must be at least num stages"
        assert world_size > 1 and num_stages > 1, "Should have more than one worker and stage"
        self.rank, self.world_size, self.num_stages, self.master_host, self.master_port = rank, world_size, num_stages, master_host, master_port

        # Initialize world structure
        self.ranks2stage = {r: self._assign_stage(r) for r in range(self.world_size)}
        self.stage2ranks = {stage: [r for r, s in self.ranks2stage.items() if s == stage] for stage in range(self.num_stages)}
        self.stage = self.ranks2stage[self.rank]
        self.is_first_stage, self.is_last_stage = self.stage == 0, self.stage == self.num_stages - 1
        self.is_master, self.is_leader = self.rank == 0, self.rank == self.get_stage_ranks(self.stage)[0]

        # Initialize global process group and store
        self.store = dist.TCPStore(self.master_host, self.master_port, self.world_size, is_master=self.is_master)
        self.global_pg = dist.init_process_group("gloo", rank=self.rank, world_size=self.world_size, store=self.store)
        self.local_pg = {}
        for stage1, stage2 in zip(range(self.num_stages-1), range(1, self.num_stages)):
            self.local_pg[(stage1, stage2)] = dist.new_group(self.stage2ranks[stage1] + self.stage2ranks[stage2])
        self.local_pg[self.stage] = dist.new_group(self.stage2ranks[self.stage], use_local_synchronization=True)
        self.prev_stage_group = self.local_pg.get((self.stage-1, self.stage), None)
        self.curr_stage_group = self.local_pg[self.stage]
        self.next_stage_group = self.local_pg.get((self.stage, self.stage+1), None)

        # Synchronize world setup
        dist.barrier()

    def get_stage_ranks(self, stage: int) -> List[int]:
        return self.stage2ranks.get(stage, [])

    def setup_step(self, step: int, queries_total: int) -> None:
        self.store.set("step", str(step))
        self.store.set("micro_steps_left", str(queries_total))

    def micro_step_done(self) -> None:
        self.store.add("micro_steps_left", -1)
        if self.micro_steps_left == 0:
            self.store.add("step", 1)

    def step_done(self, local_step: int) -> bool:
        return local_step < self.step

    @property
    def micro_steps_left(self) -> int:
        return int(self.store.get(f"micro_steps_left").decode())

    @property
    def step(self) -> int:
        return int(self.store.get(f"step").decode())

    def _assign_stage(self, rank: int) -> int: # Uniformly distribute ranks to stages
        return rank % self.num_stages

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

class SwarmComm:
    def __init__(self, model: ShardedModel, world: SwarmWorld, serializer: Serializer, shape: Tuple):
        self.model, self.world, self.serializer, self.shape = model, world, serializer, shape
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
        # print(f"Rank {self.world.rank}: Sent activations to {dst} {metadata}")

    def send_backward(self, dst: int, tensor: torch.Tensor, metadata: Metadata) -> None:
        if self.world.is_first_stage: return
        self.backward_send_queue.put((dst, tensor, metadata))
        # print(f"Rank {self.world.rank}: Sent gradients to {dst} {metadata}")

    def recv_forward(self) -> Optional[Tuple[int, DeserializedType]]:
        src, tensor, metadata = self.forward_recv_queue.get()
        # print(f"Rank {self.world.rank}: Received activations from {src} {metadata}")
        return src, tensor, metadata

    def recv_backward(self) -> Optional[Tuple[int, DeserializedType]]:
        if self.world.is_last_stage: return None
        src, tensor, metadata = self.backward_recv_queue.get()
        # print(f"Rank {self.world.rank}: Received gradients from {src} {metadata}")
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
    
    def sync_loss(self, loss: torch.Tensor) -> None:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.world.curr_stage_group)

    def _receive_loop(self, recv_queue: Queue, shape: Tuple[int, int], group: dist.ProcessGroup | None = None):
        while True:
            serialized = torch.empty(shape, device="cpu", requires_grad=True)
            src = dist.recv(serialized, group=group)
            tensor, metadata = self.serializer.deserialize(serialized)
            recv_queue.put((src, tensor, metadata))

    def _send_loop(self, send_queue: Queue, group: dist.ProcessGroup | None = None):
        while True:
            if send_queue.empty():
                time.sleep(TIMEOUT)
                continue
            dst, tensor, metadata = send_queue.get()
            serialized = self.serializer.serialize(tensor, metadata)
            dist.send(serialized, dst=dst, group=group)

def train(rank: int, world_size: int, num_stages: int, master_host: str, master_port: int):
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize store and pg
    world = SwarmWorld(rank, world_size, num_stages, master_host, master_port)

    # Setup model and data
    data_args = {"length": 100}
    model_args = {"hidden_dim": 100, "num_layers": 2}
    max_steps, num_samples, batch_size, micro_batch_size = 100, 100, 100, 10
    assert batch_size % micro_batch_size == 0, "Regular batch size must be divisible by micro batch size"
    assert (num_samples % batch_size) % micro_batch_size == 0, "Last batch size must be divisible by micro batch size"
    model = ShardedModel(Model(**model_args), world.stage, world.num_stages)
    dataloader = cycle(DataLoader(Data(**data_args), batch_size=batch_size, shuffle=True, drop_last=True))

    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3)
    loss_fn = nn.MSELoss()

    # Initialize communication
    serializer = Serializer(shape=(micro_batch_size, model_args["hidden_dim"]))
    comm = SwarmComm(model, world, serializer, serializer.shape)

    # Start training
    for step in range(1, max_steps + 1):
        # Synchronize all processes
        dist.barrier()

        # Setup step
        num_micro_steps = batch_size // micro_batch_size
        world.setup_step(step, num_micro_steps)
        optimizer.zero_grad()

        # Prepare batch batch
        batch = next(dataloader)
        batch_data = BatchData(batch)
        targets = {} # (rank, local_micro_step) -> micro_batch
        for rank in world.get_stage_ranks(0):
            micro_sampler = DistributedSampler(batch_data, rank=rank, ranks=world.get_stage_ranks(0), micro_batch_size=micro_batch_size)
            micro_dataloader = iter(DataLoader(batch_data, batch_size=micro_batch_size, sampler=micro_sampler, shuffle=False))
            for local_micro_step, micro_batch in enumerate(micro_dataloader, start=1):
                if world.is_last_stage: targets[(rank, local_micro_step)] = micro_batch["targets"].detach()
                elif world.is_first_stage and rank == world.rank: comm.load_forward_queue(local_micro_step, micro_batch["inputs"])

        # Training loop
        batch_loss = torch.tensor(0.0)
        input_output_tensors = {} # (rank, local_micro_step) -> (src, input_tensor, output_tensor)
        while not world.step_done(step):
            if comm.can_receive_forward(): # Forward
                src, input_tensor, (root, local_micro_step) = comm.recv_forward()
                output_tensor = model(input_tensor)
                input_output_tensors[(root, local_micro_step)] = (src, input_tensor, output_tensor)
                comm.send_forward(output_tensor, (root, local_micro_step))

                if world.is_last_stage: # Last stage: Compute loss and send gradients
                    loss = loss_fn(output_tensor, targets[(root, local_micro_step)])
                    batch_loss += loss.detach()
                    input_tensor_grad = model.backward(input_tensor, loss, None)
                    comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

            if comm.can_receive_backward(): # Backward
                _, output_tensor_grad, (root, local_micro_step) = comm.recv_backward()
                src, input_tensor, output_tensor = input_output_tensors[(root, local_micro_step)]
                input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
                comm.send_backward(src, input_tensor_grad, (root, local_micro_step))

                if world.is_first_stage: # First stage: Signal to world that query is done
                    world.micro_step_done()

        # Last stage: Compute batch loss
        if world.is_last_stage:
            comm.sync_loss(batch_loss)

            if world.is_leader:
                print(f"Step {step}: Loss: {batch_loss.item() / num_micro_steps:.5f}")

        # Sync and update gradients
        comm.sync_gradients()
        optimizer.step()

    # Final cleanup
    dist.destroy_process_group()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2, help="Number of workers")
    parser.add_argument("--num-stages", type=int, default=2, help="Number of stages")
    parser.add_argument("--master-host", type=str, default="localhost", help="Master hostname")
    parser.add_argument("--master-port", type=int, default=29500, help="Master port")
    args = parser.parse_args()
    assert args.world_size >= args.num_stages, "World size must be greater than or equal to number of stages"
    mp.spawn(train, args=(args.world_size, args.num_stages, args.master_host, args.master_port), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main() 