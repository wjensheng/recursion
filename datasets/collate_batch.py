import torch


def train_collate_fn(batch):
    input_, id_codes, target = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    input_, id_codes, target = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids