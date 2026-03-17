import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from icecream import ic


class Toy2dDataset(Dataset):
    def __init__(self, n_steps, n_samples):
        super().__init__()

        self.n_steps = n_steps
        self.n_samples_per_batch = n_samples

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        samples = np.random.uniform(-1.0, 1.0, (self.n_samples_per_batch, 2))
        return {
            "samples": samples.astype(np.float32),
            "sample_sdfs": np.empty((len(samples),), dtype=np.float32),
        }


class SDFDataset(Dataset):
    def __init__(self, sdf_path, n_steps, n_samples):
        super().__init__()

        data = np.load(sdf_path)

        self.samples = data["samples"]
        self.sample_sdfs = data["sample_sdfs"]

        self.n_steps = n_steps
        self.n_samples_per_batch = n_samples
        self.n_batches = len(self.samples) // n_samples

    def __len__(self):
        return self.n_steps

    def __getitem__(self, index):
        prefix = (index % self.n_batches) * self.n_samples_per_batch
        idx = np.arange(self.n_samples_per_batch) + prefix
        samples = self.samples[idx]
        sample_sdfs = self.sample_sdfs[idx]

        return {
            "samples": samples.astype(np.float32),
            "sample_sdfs": sample_sdfs.astype(np.float32),
        }


def config_dataloader(DatasetCLS, *kwargs) -> DataLoader:
    random.seed(0)
    np.random.seed(0)
    dataset = DatasetCLS(*kwargs)

    g = torch.Generator()
    g.manual_seed(0)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=g,
    )
    return dataloader


def config_toy_dataloader(n_steps, n_samples) -> DataLoader:
    return config_dataloader(Toy2dDataset, n_steps, n_samples)


def config_sdf_dataloader(sdf_path, n_steps, n_samples) -> DataLoader:
    return config_dataloader(SDFDataset, sdf_path, n_steps, n_samples)
