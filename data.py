"""DataLoader for MNIST and CIFAR10
"""
from dataclasses import dataclass

from torchvision.transforms import Resize, ToTensor, Normalize, Compose
from torchvision.datasets import MNIST, CIFAR10

from torch.utils.data import DataLoader

@dataclass
class MNISTConfig:
    dataset: str = "mnist"
    batch_size: int = 64
    image_size: int = 28
    num_channels: int = 1
    data_dir: str = ".data/"
    split: str = "train"
    num_workers: int = 4
    num_classes: int = 10


@dataclass
class CIFAR10Config:
    dataset: str = "cifar10"
    batch_size: int = 64
    image_size: int = 32
    num_channels: int = 3
    data_dir: str = ".data/"
    split: str = "train"
    num_workers: int = 4
    num_classes: int = 10


DataConfig = MNISTConfig | CIFAR10Config


def create_dataloader(config: DataConfig):
    dataset_name = config.dataset.lower()
    dataset_class = MNIST if dataset_name == "mnist" else CIFAR10
    dataset_obj = dataset_class(
        root=config.data_dir,
        train=(config.split == "train"),
        download=True,
        transform=Compose([Resize(config.image_size), ToTensor(), Normalize(0.5, 0.5)])
    )
    
    dataloader = DataLoader(
        dataset_obj,
        batch_size=config.batch_size,
        shuffle=(config.split == "train"),
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return dataloader
