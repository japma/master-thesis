import random
from collections import defaultdict

import torch
from torch.utils.data import Sampler, Dataset


class SingleClassSampler(Sampler):
    """
    A sampler that yields batches where every sample in a batch shares the
    same class label. Batches from different classes are shuffled together
    so the model sees all classes interleaved across an epoch rather than
    all of class 0, then all of class 1, etc.

    Args:
        dataset:    the dataset to sample from
        batch_size: number of samples per batch
    """

    def __init__(self, dataset: Dataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

        self.class_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.class_indices[label].append(idx)

    def __iter__(self):
        shuffled = {
            cls: torch.randperm(len(indices)).tolist()
            for cls, indices in self.class_indices.items()
        }

        batches = []
        for cls, indices in shuffled.items():
            original = self.class_indices[cls]
            shuffled_indices = [original[i] for i in shuffled[cls]]
            for i in range(0, len(shuffled_indices), self.batch_size):
                batches.append(shuffled_indices[i : i + self.batch_size])

        random.shuffle(batches)

        return iter(idx for batch in batches for idx in batch)

    def __len__(self):
        return len(self.dataset)
