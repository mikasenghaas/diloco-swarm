import random
import threading
from typing import Dict, List, Tuple
from queue import Queue

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler

class Data(Dataset):
    def __init__(self, length: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"input_ids": torch.ones(self.hidden_dim) * idx, "labels": torch.ones(self.hidden_dim) * idx}

class MicroBatchDataset(Dataset):
    def __init__(self, batch: Dict[str, torch.Tensor]):
        self.batch = batch

    def __len__(self):
        return len(self.batch["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.batch.items()}

class DistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, rank: int, ranks: List[int]):
        self.dataset = dataset
        self.rank = rank
        self.ranks = ranks
        self.num_ranks = len(ranks)
        self.idx2rank = {i: rank for i, rank in enumerate(ranks)}
        self.rank2idx = {rank: i for i, rank in enumerate(ranks)}

    def __len__(self):
        return (len(self.dataset) + self.num_ranks - self.rank2idx[self.rank] - 1) // self.num_ranks

    def __iter__(self):
        return iter([
            batch_idx for batch_idx in range(len(self.dataset)) 
            if self.idx2rank[batch_idx % self.num_ranks] == self.rank
        ])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Identity()
        self.fc2 = nn.Identity()
        self.layers = nn.ModuleList([self.fc1, self.fc2])

class ShardedModel(nn.Module):
    def __init__(self, model: nn.Module, stage: int):
        super().__init__()
        self.layers = nn.ModuleList([model.layers[stage]])
        del model

    def forward(self, input_ids: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        assert input_ids is not None or hidden_states is not None
        x = hidden_states if hidden_states is not None else input_ids
        for layer in self.layers:
            x = layer(x)
        return x

def receive(recv_queue: Queue, shape: Tuple[int, int]):
    while True:
        tensor = torch.empty(shape, device="cpu")
        src = dist.recv(tensor)
        recv_queue.put((src, tensor.detach().clone()))

def train(rank: int, world_size: int, num_stages: int, master_host: str, master_port: int):
    # Initialize store and pg
    store = dist.TCPStore(master_host, master_port, world_size, is_master=rank == 0)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, store=store)

    # Distribute ranks to stages
    stage = rank % num_stages
    store.set(f"rank{rank}", str(stage))

    num_samples = 8
    batch_size = 4
    micro_batch_size = 2
    hidden_dim = 1
    assert num_samples % batch_size == 0, "Num samples must be divisible by batch size"
    assert batch_size % micro_batch_size == 0, "Batch size must be divisible by micro batch size"
    store.set(f"total_micro_steps", str(batch_size // micro_batch_size))
    model = ShardedModel(Model(), stage)
    dataset = Data(length=num_samples, hidden_dim=hidden_dim)
    data_loader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False))

    # Start receive thread
    recv_queue = Queue()
    recv_thread = threading.Thread(
        target=receive,
        args=(recv_queue, (micro_batch_size, hidden_dim)),
        daemon=True
    )
    recv_thread.start()
    
    # Synchronize all processes
    dist.barrier()
    for step in range(1, len(data_loader) + 1):
        batch = next(data_loader)
        store.set(f"step", str(step))
        store.set(f"micro_steps", "0")
        rank_to_stage = {rank: int(store.get(f"rank{rank}").decode()) for rank in range(world_size)}
        if stage == 0:
            batch_data = MicroBatchDataset(batch)
            micro_sampler = DistributedSampler(batch_data, rank=rank, ranks=[r for r, s in rank_to_stage.items() if s == 0])
            micro_dataloader = iter(DataLoader(batch_data, batch_size=micro_batch_size, sampler=micro_sampler, shuffle=False))
            for micro_step in range(1, len(micro_dataloader) + 1):
                micro_batch = next(micro_dataloader)
                output = model(micro_batch["input_ids"])
                next_stages = [r for r, s in rank_to_stage.items() if s == 1]
                dst = random.choice(next_stages)
                dist.send(output, dst=dst)
                print(f"[Rank {rank}] Step {step}, Micro {micro_step}: Sent activations to rank {dst}")
        else:
            while not (step < int(store.get("step").decode()) ):
                if recv_queue.empty():
                    continue
                src, hidden_states = recv_queue.get()
                output = model(hidden_states=hidden_states)
                print(f"[Rank {rank}] Step {step}: Got activations from rank {src} and generated: {output}")
                recv_queue.task_done()
                store.add("micro_steps", 1)
                if int(store.get("micro_steps").decode()) == int(store.get("total_micro_steps").decode()):
                    store.set("step", str(step + 1))
        
        dist.barrier()

    # Final cleanup
    dist.destroy_process_group()

def main():
    world_size, num_stages = 4, 2
    master_addr, master_port = "localhost", 29500
    mp.spawn(train, args=(world_size, num_stages, master_addr, master_port), nprocs=world_size, join=True)

if __name__ == "__main__":
    main() 