import os
from typing import List, Literal
from datetime import datetime
from collections import defaultdict

import torch.distributed as dist

from src.config import WorldConfig

class World:
    def __init__(self, world: WorldConfig):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.rank = int(os.environ["RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.master_addr = os.environ["MASTER_ADDR"]
        self.master_port = int(os.environ["MASTER_PORT"])
        self.num_stages = world.num_stages
        assert self.world_size >= self.num_stages, "World size must be at least num stages"

        # Initialize world structure
        self.ranks2stage = defaultdict(lambda: -1, {r: self._assign_stage(r) for r in range(self.world_size)})
        self.stage2ranks = defaultdict(list, {stage: [r for r, s in self.ranks2stage.items() if s == stage] for stage in range(self.num_stages)})
        self.stage2leader = defaultdict(lambda: -1, {stage: self._assign_leader(stage) for stage in range(self.num_stages)})
        self.stage = self.ranks2stage[self.rank]
        self.is_first_stage = self.stage == 0
        self.is_last_stage = self.stage == self.num_stages - 1
        self.is_master = self.rank == 0
        self.is_leader = self.rank == self.stage2leader[self.stage]

        # Initialize process groups using the same store
        self.store = dist.TCPStore(host_name=self.master_addr, port=self.master_port+1, world_size=self.world_size, is_master=(self.rank == 0))
        self.global_pg = dist.init_process_group(backend="gloo", store=self.store, rank=self.rank, world_size=self.world_size)
        self.run_id = datetime.now().replace(second=int(datetime.now().second/10)*10).strftime("%Y%m%d_%H%M%S")
        
        # Initialize stage-specific process groups
        self.local_pg = {}
        for stage1, stage2 in zip(range(self.num_stages-1), range(1, self.num_stages)):
            self.local_pg[(stage1, stage2)] = dist.new_group(self.stage2ranks[stage1] + self.stage2ranks[stage2])
        self.local_pg[(0,self.num_stages-1)] = dist.new_group(self.stage2ranks[0] + self.stage2ranks[self.num_stages-1])
        self.local_pg[self.stage] = dist.new_group(self.stage2ranks[self.stage], use_local_synchronization=True)
        
        self.prev_stage_group = self.local_pg.get((self.stage-1, self.stage), None)
        self.curr_stage_group = self.local_pg[self.stage]
        self.next_stage_group = self.local_pg.get((self.stage, self.stage+1), None)
        self.first_last_stage_group = self.local_pg[(0, self.num_stages-1)] if self.is_first_stage or self.is_last_stage else None

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
    def next_stage_ranks(self) -> List[int]:
        return self.stage2ranks[self.stage + 1]

    @property
    def first_stage_ranks(self) -> List[int]:
        return self.stage2ranks[0]

    def _assign_stage(self, rank: int) -> int:
        return rank % self.num_stages

    def _assign_leader(self, stage: int) -> int:
        return self.stage2ranks[stage][0]

    def setup_step(self, step: int, num_micro_steps: int, type: Literal["train", "eval", "test", "sample"] = "train") -> None:
        self.store.set(f"{type}_step", str(step))
        self.store.set(f"{type}_micro_steps_left", str(num_micro_steps))

    def micro_step_done(self, type: Literal["train", "eval", "test", "sample"] = "train") -> None:
        self.store.add(f"{type}_micro_steps_left", -1)
        if self.micro_steps_left(type) == 0:
            self.store.add(f"{type}_step", 1)

    def step_done(self, local_step: int, type: Literal["train", "eval", "test", "sample"] = "train") -> bool:
        return local_step < self.step(type)

    def micro_steps_left(self, type: Literal["train", "eval", "test"] = "train") -> int:
        return int(self.store.get(f"{type}_micro_steps_left").decode())

    def step(self, type: Literal["train", "eval", "test"] = "train") -> int:
        return int(self.store.get(f"{type}_step").decode())

    def __repr__(self) -> str:
        return f"World(rank={self.rank}, stage={self.stage}, is_first_stage={self.is_first_stage}, is_last_stage={self.is_last_stage}, is_master={self.is_master}, is_leader={self.is_leader})"