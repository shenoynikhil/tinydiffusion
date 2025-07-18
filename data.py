"""DataLoader for MNIST and CIFAR10
"""
from dataclasses import dataclass

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10

from torch.utils.data import DataLoader


@dataclass
class DataConfig:
    dataset: str = "mnist"
    batch_size: int = 64
    image_size: int = 32
    data_dir: str = ".data/"
    split: str = "train"
    num_workers: int = 4


def create_dataloader(config: DataConfig):
    dataset_name = config.dataset.lower()
    dataset_class = MNIST if dataset_name == "mnist" else CIFAR10
    mean = std = (0.5,) if dataset_name == "mnist" else (0.5, 0.5, 0.5)
    dataset_obj = dataset_class(
        root=config.data_dir,
        train=(config.split == "train"),
        download=True,
        transform=Compose([Resize(config.image_size), ToTensor(), Normalize(mean, std)])
    )
    
    dataloader = DataLoader(
        dataset_obj,
        batch_size=config.batch_size,
        shuffle=(config.split == "train"),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return dataloader
