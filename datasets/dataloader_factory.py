from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn
from .default import DefaultDataset
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
from .dataset_factory import get_datasets

def make_data_loader(config):
    # train_transforms = build_transforms(config, is_train=True)
    # val_transforms = build_transforms(config, is_train=False)

    num_workers = config.dataloader.num_workers

    train_ds, valid_ds, test_ds = get_datasets(config)

    num_classes = config.model.num_classes
    
    # if config.DATALOADER.SAMPLER == 'softmax':
    #     train_loader = DataLoader(
    #         train_set, batch_size=config.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
    #         collate_fn=train_collate_fn
    #     )
    
    train_loader = DataLoader(
        train_ds, batch_size=config.train.batch_size,
        sampler=RandomIdentitySampler(train_ds, config.train.batch_size, config.dataloader.num_instance),
        num_workers=num_workers, 
        # collate_fn=train_collate_fn
    )

    
    val_loader = DataLoader(
        valid_ds, batch_size=config.val.batch_size, shuffle=False, num_workers=num_workers,
        # collate_fn=val_collate_fn
    )
    return train_loader, val_loader