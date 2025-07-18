"""
Data loading utilities for diffusion models.
Supports MNIST and CIFAR10 datasets with appropriate preprocessing.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional
import lightning as L


class DiffusionDataModule(L.LightningDataModule):
    """Lightning DataModule for diffusion model training."""
    
    def __init__(
        self,
        dataset: str = "mnist",
        batch_size: int = 64,
        num_workers: int = 4,
        data_dir: str = "./data",
        image_size: int = 32,
        normalize: bool = True,
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.image_size = image_size
        self.normalize = normalize
        
        # Define transforms based on dataset
        self.setup_transforms()
    
    def setup_transforms(self):
        """Setup transforms for the specified dataset."""
        transform_list = []
        
        if self.dataset == "mnist":
            # MNIST is grayscale, resize to desired size
            transform_list.extend([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
            if self.normalize:
                # Normalize to [-1, 1] range for diffusion models
                transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        elif self.dataset == "cifar10":
            # CIFAR10 is RGB, resize if needed
            if self.image_size != 32:
                transform_list.append(transforms.Resize(self.image_size))
            transform_list.append(transforms.ToTensor())
            if self.normalize:
                # Normalize to [-1, 1] range for diffusion models
                transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        
        self.transform = transforms.Compose(transform_list)
    
    def prepare_data(self):
        """Download data if needed."""
        if self.dataset == "mnist":
            torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
            torchvision.datasets.MNIST(self.data_dir, train=False, download=True)
        elif self.dataset == "cifar10":
            torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(self.data_dir, train=False, download=True)
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        if self.dataset == "mnist":
            self.train_dataset = torchvision.datasets.MNIST(
                self.data_dir, train=True, transform=self.transform
            )
            self.val_dataset = torchvision.datasets.MNIST(
                self.data_dir, train=False, transform=self.transform
            )
        elif self.dataset == "cifar10":
            self.train_dataset = torchvision.datasets.CIFAR10(
                self.data_dir, train=True, transform=self.transform
            )
            self.val_dataset = torchvision.datasets.CIFAR10(
                self.data_dir, train=False, transform=self.transform
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def get_dataset_info(dataset: str) -> dict:
    """Get dataset information."""
    info = {
        "mnist": {
            "channels": 1,
            "height": 28,
            "width": 28,
            "num_classes": 10,
            "name": "MNIST"
        },
        "cifar10": {
            "channels": 3,
            "height": 32,
            "width": 32,
            "num_classes": 10,
            "name": "CIFAR-10"
        }
    }
    return info.get(dataset.lower(), {})


def create_dataloader(
    dataset: str = "mnist",
    batch_size: int = 64,
    image_size: int = 32,
    data_dir: str = "./data",
    split: str = "train"
) -> Tuple[DataLoader, dict]:
    """
    Create a simple dataloader without Lightning.
    
    Args:
        dataset: Dataset name ('mnist' or 'cifar10')
        batch_size: Batch size
        image_size: Target image size
        data_dir: Data directory
        split: 'train' or 'test'
    
    Returns:
        DataLoader and dataset info
    """
    # Setup transforms
    transform_list = [transforms.ToTensor()]
    
    if dataset.lower() == "mnist":
        if image_size != 28:
            transform_list.insert(0, transforms.Resize(image_size))
        # Normalize to [-1, 1]
        transform_list.append(transforms.Normalize((0.5,), (0.5,)))
        
        # Load dataset
        dataset_class = torchvision.datasets.MNIST
        
    elif dataset.lower() == "cifar10":
        if image_size != 32:
            transform_list.insert(0, transforms.Resize(image_size))
        # Normalize to [-1, 1]
        transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        
        # Load dataset
        dataset_class = torchvision.datasets.CIFAR10
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    transform = transforms.Compose(transform_list)
    
    # Create dataset
    is_train = (split == "train")
    dataset_obj = dataset_class(
        root=data_dir,
        train=is_train,
        download=True,
        transform=transform
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=4,
        pin_memory=True
    )
    
    info = get_dataset_info(dataset)
    if image_size != info.get("height", image_size):
        info["height"] = image_size
        info["width"] = image_size
    
    return dataloader, info


if __name__ == "__main__":
    # Example usage
    print("Testing data loading...")
    
    # Test with Lightning DataModule
    print("\n=== Testing Lightning DataModule ===")
    dm = DiffusionDataModule(dataset="mnist", batch_size=32, image_size=32)
    dm.prepare_data()
    dm.setup()
    
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, labels = batch
    
    print(f"MNIST batch shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Labels shape: {labels.shape}")
    
    # Test CIFAR10
    dm_cifar = DiffusionDataModule(dataset="cifar10", batch_size=32)
    dm_cifar.prepare_data()
    dm_cifar.setup()
    
    cifar_loader = dm_cifar.train_dataloader()
    batch = next(iter(cifar_loader))
    images, labels = batch
    
    print(f"\nCIFAR10 batch shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]")
    
    # Test simple dataloader
    print("\n=== Testing Simple DataLoader ===")
    loader, info = create_dataloader("mnist", batch_size=16, image_size=28)
    batch = next(iter(loader))
    images, labels = batch
    
    print(f"Dataset info: {info}")
    print(f"Batch shape: {images.shape}")
    print(f"Value range: [{images.min():.3f}, {images.max():.3f}]") 
