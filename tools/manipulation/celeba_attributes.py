from collections import defaultdict
from pathlib import Path

import fire
import numpy as np
import torch

from editing.config import CELEBA_ATTRS


class Worker:
    def __init__(self, record_path="./data/interfacegan_dataset.npz", device="cuda"):
        records = np.load(record_path)
        # 500000x40
        self.logit_bank = torch.from_numpy(records["logit"]).to(device, torch.float)
        # 500000x512
        self.w_bank = torch.from_numpy(records["w"]).to(device, torch.float)
        # 40
        self.std_bank = self.logit_bank.std(dim=0)

    def random_samples(self, attr_id, num_samples=1000, sample_range=(0.25, 0.75)):
        torch.manual_seed(0)
        num_total = len(self.logit_bank)
        index_pool = self.logit_bank[:, attr_id].topk(num_total, largest=False)[1][
                     int(num_total * sample_range[0]): int(num_total * sample_range[1])]
        torch.manual_seed(0)
        return index_pool[torch.randperm(len(index_pool))[:num_samples]]


def generate_examples():
    worker = Worker()
    examples = defaultdict(list)
    for attr_id in range(40):
        for level in range(10):
            examples[attr_id].append(worker.w_bank[worker.random_samples(attr_id, 100, (level / 10, level / 10 + 0.1))])
    torch.save(examples, "./data/attribute_examples.pt")


def sample(attr_id, num_samples, sample_range=(0.25, 0.75), save_path=None):
    assert num_samples < (sample_range[1] - sample_range[0]) * 500000
    worker = Worker()
    ws = worker.w_bank[worker.random_samples(attr_id, num_samples, sample_range)]

    default_save_name = f"{CELEBA_ATTRS[attr_id]}_{sample_range[0]}-{sample_range[1]}_{num_samples}.w"
    if save_path is None:
        save_path = default_save_name
    else:
        save_path = Path(save_path)
        if save_path.is_dir():
            save_path = save_path / default_save_name

    torch.save(ws, save_path)


if __name__ == '__main__':
    fire.Fire()
