import random
from torch.utils.data import Sampler

class CurriculumBatchSampler(Sampler):
    """
    Yields full batches of indices sorted by difficulty (easy -> medium -> hard).
    Vectorized batching eliminates PyTorch single-item indexing overhead.
    """
    def __init__(self, hf_dataset, batch_size, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.easy_idx = []
        self.medium_idx = []
        self.hard_idx = []

        if hasattr(hf_dataset, "dataset"):
            hf_dataset = hf_dataset.dataset

        if hasattr(hf_dataset, "column_names") and "difficulty" in hf_dataset.column_names:
            diffs = hf_dataset["difficulty"]
            for i, diff in enumerate(diffs):
                if diff == "easy":
                    self.easy_idx.append(i)
                elif diff == "medium":
                    self.medium_idx.append(i)
                else:
                    self.hard_idx.append(i)
        else:
            for i in range(len(hf_dataset)):
                item = hf_dataset[i]
                diff = item.get("difficulty", "medium") if isinstance(item, dict) else "medium"
                if diff == "easy":
                    self.easy_idx.append(i)
                elif diff == "medium":
                    self.medium_idx.append(i)
                else:
                    self.hard_idx.append(i)

    def __iter__(self):
        random.shuffle(self.easy_idx)
        random.shuffle(self.medium_idx)
        random.shuffle(self.hard_idx)

        all_indices = self.easy_idx + self.medium_idx + self.hard_idx
        batch = []
        for idx in all_indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        total = len(self.easy_idx) + len(self.medium_idx) + len(self.hard_idx)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size
