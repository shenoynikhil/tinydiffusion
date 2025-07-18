import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data import DataConfig, create_dataloader


def test_mnist_loading():
    config = DataConfig(dataset="mnist", batch_size=2)
    loader = create_dataloader(config)
    batch = next(iter(loader))
    assert batch[0].shape == (2, 1, 32, 32)


def test_cifar10_loading():
    config = DataConfig(dataset="cifar10", batch_size=2)
    loader = create_dataloader(config)
    batch = next(iter(loader))
    assert batch[0].shape == (2, 3, 32, 32)
