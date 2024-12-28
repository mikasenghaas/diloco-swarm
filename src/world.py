import os
from typing import List, Literal, Dict, Any, Iterator
from datetime import datetime
from collections import defaultdict

import torch.distributed as dist

from src.config import SwarmConfig

class World:
    def __init__(self, swarm: SwarmConfig):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        self.num_stages = swarm.num_stages
        assert self.world_size >= self.num_stages, "World size must be at least num stages"

        # Initialize world structure
        self.ranks2stage = defaultdict(lambda: -1, {r: self._assign_stage(r) for r in range(self.world_size)})
        self.stage2ranks = defaultdict(list, {stage: [r for r, s in self.ranks2stage.items() if s == stage] for stage in range(self.num_stages)})
        self.stage2leader = defaultdict(lambda: -1, {stage: self._assign_leader(stage) for stage in range(self.num_stages)})
        self.stage = self.ranks2stage[self.rank]
        self.is_first_stage = self.stage == 0
        self.is_last_stage = self.stage == self.num_stages - 1
        self.is_leader = self.rank == self.stage2leader[self.stage]
        self.is_master = self.is_leader and self.is_last_stage

        # Initialize process groups using the same store
        self.store = dist.TCPStore(host_name=self.master_addr, port=self.master_port+1, world_size=self.world_size, is_master=(self.rank == 0))
        self.global_pg = dist.init_process_group(backend="gloo", store=self.store, rank=self.rank, world_size=self.world_size)
        
        # Initialize stage-specific process groups
        self.local_pg = {(stage, stage+1): dist.new_group(self.stage2ranks[stage] + self.stage2ranks[stage+1]) for stage in range(self.num_stages-1)}
        self.prev_stage_group = self.local_pg.get((self.stage-1, self.stage), None)
        self.next_stage_group = self.local_pg.get((self.stage, self.stage+1), None)
        self.curr_stage_group = dist.new_group(self.stage2ranks[self.stage], use_local_synchronization=True)
        self.first_last_stage_group = dist.new_group(list(set(self.stage2ranks[0] + self.stage2ranks[self.num_stages-1])), use_local_synchronization=True)

        # Synchronize world setup
        dist.barrier()

    @property
    def next_stage_leader(self) -> int:
        return self.stage2leader[self.stage + 1]

    @property
    def prev_stage_leader(self) -> int:
        return self.stage2leader[self.stage - 1]

    @property
    def first_stage_leader(self) -> int:
        return self.stage2leader[0]

    @property
    def last_stage_leader(self) -> int:
        return self.stage2leader[self.num_stages - 1]

    @property
    def has_next_stage(self) -> bool:
        return self.stage2ranks[self.stage + 1] != []

    @property
    def has_prev_stage(self) -> bool:
        return self.stage2ranks[self.stage - 1] != []

    @property
    def first_stage_ranks(self) -> List[int]:
        return self.stage2ranks[0]

    def setup_step(self, step: int, num_micro_steps: int, type: str = "train") -> None:
        if self.is_master:
            self.store.set(f"{type}_step", str(step))
            self.store.set(f"{type}_micro_steps_left", str(num_micro_steps))
        dist.barrier()

    def micro_step_done(self, type: str = "train") -> None:
        self.store.add(f"{type}_micro_steps_left", -1)
        if self.micro_steps_left(type) == 0:
            self.step_done(type)

    def step_done(self, type: str = "train") -> None:
        self.store.add(f"{type}_step", 1)

    def is_step_done(self, local_step: int, type: str = "train") -> bool:
        return local_step < self.step(type)

    def micro_steps_left(self, type: str = "train") -> int:
        return int(self.store.get(f"{type}_micro_steps_left").decode())

    def step(self, type: str = "train") -> int:
        return int(self.store.get(f"{type}_step").decode())

    def _assign_stage(self, rank: int) -> int:
        return rank % self.num_stages

    def _assign_leader(self, stage: int) -> int:
        return self.stage2ranks[stage][0]

    def __repr__(self) -> str:
        return f"World(rank={self.rank}, stage={self.stage}, is_first_stage={self.is_first_stage}, is_last_stage={self.is_last_stage}, is_master={self.is_master}, is_leader={self.is_leader})"

    def __iter__(self) -> Iterator[int]:
        return iter(self.__dict__().items())

    def __dict__(self) -> Dict[str, Any]:
        return {
            "local_rank": self.local_rank,
            "rank": self.rank,
            "local_world_size": self.local_world_size,
            "world_size": self.world_size,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "num_stages": self.num_stages,
            "is_master": self.is_master,
            "is_leader": self.is_leader
        }