from typing import Dict, List

import torch
from torch.utils.data import Sampler, Dataset

class BatchData(Dataset):
    def __init__(self, batch: Dict[str, torch.Tensor]):
        self.batch = batch

    def __len__(self) -> int:
        return len(next(iter(self.batch.values())))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {k: v[idx] if v is not None else v for k, v in self.batch.items()}

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