import torch
from itertools import accumulate


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(lengths)
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in (accumulate(lengths), lengths)]
