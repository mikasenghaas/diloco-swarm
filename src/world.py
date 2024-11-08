import torch.distributed as dist

class World:
    def __init__(self, local_rank: int, world_size: int):
        self.local_rank = local_rank
        self.world_size = world_size
        self.group = None

        if world_size > 1:
            dist.init_process_group(backend="nccl")
            self.group = dist.new_group(list(range(self.world_size)))

    def get_next_rank(self) -> int:
        return None if self.local_rank == self.world_size - 1 else (self.local_rank + 1) % self.world_size

    def get_prev_rank(self) -> int:
        return None if self.local_rank == 0 else (self.local_rank - 1) % self.world_size

    @property
    def is_first_stage(self) -> bool:
        return self.local_rank == 0

    @property
    def is_last_stage(self) -> bool:
        return self.local_rank == self.world_size - 1

    def __repr__(self) -> str:
        return f"World(local_rank={self.local_rank}, world_size={self.world_size}, is_first_stage={self.is_first_stage}, is_last_stage={self.is_last_stage})"

