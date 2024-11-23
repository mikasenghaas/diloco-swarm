import time
import random
import threading
from queue import Queue
from typing import Dict, List, Tuple
from itertools import cycle as cycle_iter

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader, Sampler

TIMEOUT = 0.01

class Data(Dataset):
    def __init__(self, length: int):
        self.length = length
        self.x = torch.linspace(-5, 5, length).float().reshape(-1, 1).requires_grad_(True)
        self.y = self.x ** 2 # learn quadratic function

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {"inputs": self.x[idx], "labels": self.y[idx]}

class MicroBatchDataset(Dataset):
    def __init__(self, batch: Dict[str, torch.Tensor]):
        self.batch = batch

    def __len__(self):
        return len(self.batch["inputs"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.batch.items()}

class DistributedSampler(Sampler):
    def __init__(self, dataset: Dataset, rank: int, ranks: List[int], micro_batch_size: int):
        self.dataset = dataset
        self.rank = rank
        self.ranks = ranks
        self.num_ranks = len(ranks)
        self.idx2rank = {i: rank for i, rank in enumerate(ranks)}
        self.rank2idx = {rank: i for i, rank in enumerate(ranks)}
        self.micro_batch_size = micro_batch_size

    def __len__(self):
        total_samples = len(self.dataset)
        samples_per_full_round = self.num_ranks * self.micro_batch_size
        
        full_rounds = total_samples // samples_per_full_round
        samples_from_full_rounds = full_rounds * self.micro_batch_size
        
        remaining_samples = total_samples % samples_per_full_round
        rank_start_in_last_round = self.rank2idx[self.rank] * self.micro_batch_size
        remaining_for_rank = max(0, min(self.micro_batch_size, remaining_samples - rank_start_in_last_round))
        
        return samples_from_full_rounds + remaining_for_rank

    def __iter__(self):
        for batch_start in range(0, len(self.dataset), self.num_ranks * self.micro_batch_size):
            rank_start = batch_start + (self.rank2idx[self.rank] * self.micro_batch_size)
            for idx in range(rank_start, min(rank_start + self.micro_batch_size, len(self.dataset))):
                yield idx

class Model(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embed = nn.Linear(1, hidden_dim)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(2)
        ])
        self.head = nn.Linear(hidden_dim, 1)

class ShardedModel(nn.Module):
    def __init__(self, model: nn.Module, stage: int, num_stages: int):
        super().__init__()
        self.embed = model.embed if stage == 0 else nn.Identity()
        self.stage = stage
        self.num_stages = num_stages
        self.blocks = nn.ModuleList([model.blocks[i] for i in self.distribute_layers(len(model.blocks))])
        self.head = model.head if stage == num_stages - 1 else nn.Identity()
        del model

    def distribute_layers(self, num_layers):
        layers_per_gpu = [num_layers // self.num_stages + (1 if i < num_layers % self.num_stages else 0) for i in range(self.num_stages)]
        start_layer = sum(layers_per_gpu[:self.stage])
        return list(range(start_layer, start_layer + layers_per_gpu[self.stage]))

    def forward(self, input_ids: torch.Tensor | None = None, hidden_states: torch.Tensor | None = None) -> torch.Tensor:
        assert input_ids is not None or hidden_states is not None
        x = hidden_states if hidden_states is not None else self.embed(input_ids)
        for layer in self.blocks:
            x = layer(x)
        return self.head(x)

    def backward(self, input_tensor, output_tensor, output_tensor_grad):
        if input_tensor is not None: input_tensor.retain_grad()
        if output_tensor_grad is None:
            output_tensor_grad = torch.ones_like(output_tensor, memory_format=torch.preserve_format)
        torch.autograd.backward(output_tensor, grad_tensors=output_tensor_grad)
        return input_tensor.grad if input_tensor is not None else None

def start_receiving(recv_queue: Queue, shape: Tuple[int, int], group: dist.ProcessGroup | None = None):
    while True:
        serialized = torch.empty(shape, device="cpu", requires_grad=True)
        src = dist.recv(serialized, group=group)
        (root, batch_idx), tensor = deserialize(serialized, 2)
        recv_queue.put((src, root, batch_idx, tensor))

def start_sending(send_queue: Queue, group: dist.ProcessGroup | None = None):
    while True:
        if send_queue.empty():
            time.sleep(TIMEOUT)
            continue
        dst, root, batch_idx, tensor = send_queue.get()
        serialized = serialize(tensor, root, batch_idx)
        dist.send(serialized, dst=dst, group=group)

def serialize(tensor: torch.Tensor, *args: int) -> torch.Tensor:
    assert len(args) <= tensor.numel(), "Can't fit metadata in tensor's first dim" # can use all dims later
    metadata = torch.empty(tensor.numel())
    for i, arg in enumerate(args):
        metadata[i] = float(arg)
    metadata = metadata.reshape(tensor.shape)
    return torch.cat([metadata.unsqueeze(0), tensor.unsqueeze(0)], dim=0)

def deserialize(combined: torch.Tensor, num_args: int) -> Tuple[Tuple[int, ...], torch.Tensor]:
    # First row iss metadata, rest is tensor
    metadata = combined[0].flatten()
    tensor = combined[1:]   # Get tensor from remaining dimensions
    args = []
    for i in range(num_args):
        args.append(int(metadata[i].item()))
    # Convert metadata tensor to tuple of ints
    return tuple(args), tensor.squeeze(0)

def sync_tensor(tensor: torch.Tensor, stage_group: dist.ProcessGroup):
    """Synchronize tensor between workers in the same pipeline stage."""
    tensor.div_(dist.get_world_size(stage_group))

def sync_gradients(model: nn.Module, stage_group: dist.ProcessGroup):
    """Synchronize gradients between workers in the same pipeline stage."""
    for param in model.parameters():
        if param.grad is None: param.grad = torch.zeros_like(param)
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=stage_group)

def train(rank: int, world_size: int, num_stages: int, master_host: str, master_port: int):
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Initialize store and pg
    store = dist.TCPStore(master_host, master_port, world_size, is_master=rank == 0)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, store=store)

    # Distribute ranks to stages
    stage = rank % num_stages
    store.set(f"rank{rank}", str(stage))
    rank_to_stage = {rank: int(store.get(f"rank{rank}").decode()) for rank in range(world_size)}

    # Initialize overlapping stage groups
    groups = {}
    for stage1, stage2 in zip(range(num_stages-1), range(1, num_stages)):
        stage_ranks = [r for r, s in rank_to_stage.items() if s in (stage1, stage2)]
        groups[(stage1, stage2)] = dist.new_group(stage_ranks)

    # Local peers
    peers = {}
    peers["prev"] = groups.get((stage-1, stage), None)
    peers["self"] = dist.new_group([r for r, s in rank_to_stage.items() if s == stage], use_local_synchronization=True)
    peers["next"] = groups.get((stage, stage+1), None)
    tmp = {k: dist.get_process_group_ranks(v) if v is not None else None for k, v in peers.items()}
    # print(f"[Rank {rank}] Peers: {tmp}")
    dist.barrier()

    # Setup
    max_steps = 10
    num_samples = 100
    batch_size = 100
    micro_batch_size = 100
    hidden_dim = 100
    assert batch_size % micro_batch_size == 0, "Regular batch size must be divisible by micro batch size"
    assert (num_samples % batch_size) % micro_batch_size == 0, "Last batch size must be divisible by micro batch size"
    model = ShardedModel(Model(hidden_dim=hidden_dim), stage, num_stages)
    dataset = Data(length=num_samples)
    data_loader = cycle_iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))

    # Compute tensor shapes (+metadata)
    shape = (micro_batch_size, hidden_dim) # without metadata
    combined_shape = (2, micro_batch_size, hidden_dim) # with metadata
    # if rank == 0:
    #     inputs = next(data_loader)["inputs"]
    #     print(inputs.shape)
    #     output = model.forward(input_ids=inputs)
    #     print(output.shape)
    #     src, batch_idx = 0, 1
    #     serialized = serialize(output, src, batch_idx)
    #     print(serialized.shape)
    #     (src, batch_idx), tensor = deserialize(serialized, 2)
    #     print(tensor, src, batch_idx)

    # Start bi-directional communication threads
    forward_send_queue = Queue()
    backward_send_queue = Queue()
    forward_recv_queue = Queue()
    backward_recv_queue = Queue()
    threading.Thread(target=start_receiving, args=(forward_recv_queue, combined_shape, peers["prev"]), daemon=True).start()
    threading.Thread(target=start_receiving, args=(backward_recv_queue, combined_shape, peers["next"]), daemon=True).start() # Alternative: just one thread and distingish based on src rank
    threading.Thread(target=start_sending, args=(forward_send_queue, peers["next"]), daemon=True).start()
    threading.Thread(target=start_sending, args=(backward_send_queue, peers["prev"]), daemon=True).start()


    # Initialize optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    loss_fn = nn.MSELoss()
    
    # Synchronize all processes
    dist.barrier()
    for step in range(1, max_steps + 1):
        batch = next(data_loader)
        optimizer.zero_grad()
        total_micro_steps = batch["inputs"].shape[0] // micro_batch_size
        store.set(f"step", str(step))
        store.set(f"micro_steps", str(0))
        store.set(f"total_micro_steps", str(total_micro_steps))
        rank_to_stage = {rank: int(store.get(f"rank{rank}").decode()) for rank in range(world_size)}

        # Get shared micro batches (to get labels on last rank)
        batch_data = MicroBatchDataset(batch)
        stage0_ranks = [r for r, s in rank_to_stage.items() if s == 0]
        targets = {} # (root, micro_step) -> micro_batch
        target_list = []
        for r in stage0_ranks:
            micro_sampler = DistributedSampler(batch_data, rank=r, ranks=stage0_ranks, micro_batch_size=micro_batch_size)
            micro_dataloader = iter(DataLoader(batch_data, batch_size=micro_batch_size, sampler=micro_sampler, shuffle=False))
            for micro_step, micro_batch in enumerate(micro_dataloader, start=1):
                target = micro_batch["labels"].detach()
                target_list.append(target)
                targets[(r, micro_step)] = target

        # Forward pass
        if stage == 0: # first stage
            micro_sampler = DistributedSampler(batch_data, rank=rank, ranks=[r for r, s in rank_to_stage.items() if s == 0], micro_batch_size=micro_batch_size)
            micro_dataloader = iter(DataLoader(batch_data, batch_size=micro_batch_size, sampler=micro_sampler, shuffle=False))
            output_tensors = {}
            for micro_step, micro_batch in enumerate(micro_dataloader, start=1):
                output_tensor = model(input_ids=micro_batch["inputs"])
                output_tensors[(rank, micro_step)] = output_tensor
                dst = random.choice([r for r, s in rank_to_stage.items() if s == 1])
                forward_send_queue.put((dst, rank, micro_step, output_tensor))
                # print(f"[Rank {rank}] Step {step}: Sent activations (Micro Step {micro_step}, Root {rank}, Dst {dst})")

            while not (step < int(store.get("step").decode())):
                if backward_recv_queue.empty():
                    time.sleep(TIMEOUT)
                    continue
                src, root, micro_step, output_tensor_grad = backward_recv_queue.get()
                assert root == rank, "Gradients must have originated at rank"
                # print(f"[Rank {rank}] Step {step}: Received gradients (Micro Step {micro_step}, Root {root}, Src {src})")
                model.backward(None, output_tensors[(root, micro_step)], output_tensor_grad)
                store.add("micro_steps", 1)

                if int(store.get("micro_steps").decode()) == int(store.get("total_micro_steps").decode()):
                    store.set("step", str(step + 1))
        elif stage == num_stages - 1: # last stage
            batch_loss = torch.tensor(0.0, device="cpu")
            while not (step < int(store.get("step").decode()) ):
                if forward_recv_queue.empty():
                    time.sleep(TIMEOUT)
                    continue
                src, root, micro_step, input_tensor = forward_recv_queue.get()
                # print(f"[Rank {rank}] Step {step}: Received activations (Micro Step {micro_step}, Root {root}, Src {src})")
                output_tensor = model(hidden_states=input_tensor) # preds for micro_step
                loss = loss_fn(output_tensor, targets[(root, micro_step)])
                # print(f"[Rank {rank}] Step {step}: Computed loss")
                
                input_tensor_grad = model.backward(input_tensor, loss, None)
                backward_send_queue.put((src, root, micro_step, input_tensor_grad))
                # print(f"[Rank {rank}] Step {step}: Sent gradients (Micro Step {micro_step}, Root {root}, Dst {src})")

                batch_loss += loss.detach()

            # Sum loss across workers and divide by total number of micro batches
            dist.barrier(group=peers["self"])
            dist.all_reduce(batch_loss, op=dist.ReduceOp.SUM, group=peers["self"])
            batch_loss = batch_loss / total_micro_steps

            if rank == dist.get_process_group_ranks(peers["self"])[0]:
                print(f"Step {step}: Loss: {batch_loss.item()}")
        else: # middle stages
            output_tensors = {}
            while not (step < int(store.get("step").decode())):
                if forward_recv_queue.empty() and backward_recv_queue.empty():
                    time.sleep(TIMEOUT)
                    continue
                if not forward_recv_queue.empty():
                    src, root, micro_step, input_tensor = forward_recv_queue.get()
                    # print(f"[Rank {rank}] Step {step}: Received activations (Micro Step {micro_step}, Root {root}, Src {src})")
                    output_tensor = model(hidden_states=input_tensor) # preds for micro_step
                    dst = random.choice([r for r, s in rank_to_stage.items() if s == stage + 1]) # next stage
                    output_tensors[(root, micro_step)] = (src, input_tensor, output_tensor)
                    forward_send_queue.put((dst, root, micro_step, output_tensor))
                    # print(f"[Rank {rank}] Step {step}: Sent activations to rank {dst} (Micro Step {micro_step}, Root {root})")
                if not backward_recv_queue.empty():
                    src, root, micro_step, output_tensor_grad = backward_recv_queue.get()
                    # print(f"[Rank {rank}] Step {step}: Received gradients (Micro Step {micro_step}, Root {root}, Src {src})")
                    (src, input_tensor, output_tensor) = output_tensors[(root, micro_step)]
                    input_tensor_grad = model.backward(input_tensor, output_tensor, output_tensor_grad)
                    backward_send_queue.put((src, root, micro_step, input_tensor_grad))
                    # print(f"[Rank {rank}] Step {step}: Sent gradients (Micro Step {micro_step}, Root {root}, Dst {src})")


        # Sync gradients
        # print(f"[Rank {rank}] Syncing gradients for step {step}")
        sync_gradients(model, peers["self"])

        # Backward pass
        optimizer.step()
        dist.barrier()

    # Final cleanup
    dist.destroy_process_group()

def main():
    world_size, num_stages = 6, 3
    assert world_size >= num_stages, "World size must be greater than or equal to number of stages"
    master_addr, master_port = "localhost", 29500
    mp.spawn(train, args=(world_size, num_stages, master_addr, master_port), nprocs=world_size, join=True)

if __name__ == "__main__":
    main() 