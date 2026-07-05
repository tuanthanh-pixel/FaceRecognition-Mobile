from torch.utils.data import DataLoader

from configs import config
from data.dataset import CASIADataset
from data.transform import train_transform


def get_train_loader():

    dataset = CASIADataset(
        root_dir=config.CASIA_PATH,
        transform=train_transform
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == "cuda")
    )

    return train_loader