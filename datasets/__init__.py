from .dataset_DFEW import DFEWDataset
from .dataset_daisee import DaiseeDataset

from torch.utils import data
from torchsampler import ImbalancedDatasetSampler


def create_dataloader(args, mode):
    """create dataloader according to args and training/testing mode

    Args:
        args
        mode: String("train" or "test")

    Returns:
        dataloader
    """
    # dataset = DFEWDataset(args, mode)
    dataset = DaiseeDataset(args, mode)
    dataloader = None

    # return train_dataset or test_dataset according to the mode
    if mode == "train":
        dataloader = data.DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     # sampler=ImbalancedDatasetSampler(dataset),
                                     shuffle=True,
                                     num_workers=args.workers,
                                     pin_memory=True,
                                     drop_last=True)
    elif mode == "test":
        dataloader = data.DataLoader(dataset,
                                     batch_size=args.batch_size,
                                     shuffle=False,
                                     num_workers=args.workers,
                                     pin_memory=True)
    return dataloader
