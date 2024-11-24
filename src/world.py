import os
import time
from typing import List
from datetime import datetime

import torch
import torch.distributed as dist

from src.config import WorldConfig

class World:
    def __init__(self, local_rank: int, world_size: int, device: torch.device, debug: bool = False):
        os.environ["OMP_NUM_THREADS"] = "1"
        self.num_threads = int(os.environ.get("OMP_NUM_THREADS", 1))
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device

        if world_size > 1 and not debug:
            dist.init_process_group(backend="nccl", device_id=device)
            dist.barrier()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def next_rank(self) -> int:
        return None if self.local_rank == self.world_size - 1 else (self.local_rank + 1) % self.world_size

    def prev_rank(self) -> int:
        return None if self.local_rank == 0 else (self.local_rank - 1) % self.world_size

    @property
    def first_stage(self) -> int:
        return 0

    @property
    def last_stage(self) -> int:
        return self.world_size - 1

    @property
    def is_first_stage(self) -> bool:
        return self.local_rank == self.first_stage

    @property
    def is_last_stage(self) -> bool:
        return self.local_rank == self.last_stage

    def __repr__(self) -> str:
        return f"World(local_rank={self.local_rank}, world_size={self.world_size}, is_first_stage={self.is_first_stage}, is_last_stage={self.is_last_stage}, device={self.device})"

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "local_rank": self.local_rank,
            "world_size": self.world_size,
            "device": self.device,
        }

    def __iter__(self):
        for key, value in self.to_dict().items():
            yield key, value

class SwarmWorld:
    def __init__(self, world: WorldConfig):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        self.num_stages = world.num_stages

        assert self.world_size >= self.num_stages, "World size must be at least num stages"
        assert self.world_size > 1 and self.num_stages > 1, "Should have more than one worker and stage"

        # Initialize world structure
        self.ranks2stage = {r: self._assign_stage(r) for r in range(self.world_size)}
        self.stage2ranks = {stage: [r for r, s in self.ranks2stage.items() if s == stage] for stage in range(self.num_stages)}
        self.stage = self.ranks2stage[self.rank]
        self.is_first_stage = self.stage == 0
        self.is_last_stage = self.stage == self.num_stages - 1
        self.is_master = self.rank == 0
        self.is_leader = self.rank == self.get_stage_ranks(self.stage)[0]

        # Initialize process groups using the same store
        self.store = dist.TCPStore(
            host_name=self.master_addr,
            port=self.master_port+1, # ???
            world_size=self.world_size,
            is_master=(self.rank == 0)
        )
        self.global_pg = dist.init_process_group(backend="gloo", store=self.store, rank=self.rank, world_size=self.world_size)
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize stage-specific process groups
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

    def _assign_stage(self, rank: int) -> int:
        return rank % self.num_stages

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
        return int(self.store.get("micro_steps_left").decode())

    @property
    def step(self) -> int:
        return int(self.store.get("step").decode())
