"""
Implementation based on, 
Mean Flows for One-Step Generative Modelling, Geng et al 2025.
"""

from dataclasses import dataclass
import os
import sys

import numpy as np
import torch
from torch import Tensor
import lightning as L
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint

from dit import DiTMeanFlow
from data import create_dataloader
from torchvision.utils import make_grid, save_image


@dataclass
class MeanFlowConfig:
    # mean flow args
    interpolation_type: str = "linear"

    # DiT args
    patch_size: int = 8
    num_channels: int = 3
    image_dim: int = 32
    depth: int = 12
    num_heads: int = 6
    hidden_size: int = 384
    qkv_bias: bool = True
    mlp_ratio: float = 2.0
    num_classes: int = 10

    # optimizer args
    lr: float = 3.e-4

    # evaluation args
    evaluate_every_n_epochs: int = 0
    save_path: str = "generated_images"
    normalize: bool = True


class MeanFlow(L.LightningModule):
    """
    Note: Compared to the flow-matching implementation
    t=0 (data distribution) and t=1 (prior distribution). 
    
    TODO: Reverse this.
    """
    def __init__(self, cfg: MeanFlowConfig):
        super().__init__()
        self.cfg = cfg

        self.network = DiTMeanFlow(
            input_size=self.cfg.image_dim,
            patch_size=self.cfg.patch_size,
            in_channels=self.cfg.num_channels,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            hidden_size=self.cfg.hidden_size,
            mlp_ratio=self.cfg.mlp_ratio,
            num_classes=self.cfg.num_classes,
        )

    def training_step(self, batch, batch_idx):
        """Algorithm 1, Page 5 from MeanFlows"""
        x1, y = batch # image: (b, c, h, w), and label: (b,)
        x0 = torch.randn_like(x1) # gaussian prior

        # sample t and r
        t = torch.rand(x1.shape[0],).to(self.device)
        r = torch.rand_like(t)
        
        # linearly interpolate (can be something more complex)
        # and compute conditional velocity field
        t_reshaped = rearrange(t, "b -> b 1 1 1")
        xt = x1 * (1 - t_reshaped) + x0 * t_reshaped
        v = x0 - x1

        # run DiT to compute predicted velocity field
        y_ = torch.ones_like(t) * y
        u, dudt = torch.func.jvp(
            self.network.forward, # function
            (xt, t, r, y_), # inputs to function
            (v, torch.ones_like(t), torch.zeros_like(t), y_)
        )

        # Algorithm 1: modified target based on instantaneous velocity
        u_tgt = v - rearrange((t - r), "b -> b 1 1") * dudt
        u_tgt = u_tgt.detach() # stop grad to disable gradients through jvp

        # compute loss between ut and vt, matching conditional velocity field
        loss = torch.mean((u - u_tgt) ** 2)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)

    def on_train_epoch_end(self):
        evaluate_condition = (
            self.cfg.evaluate_every_n_epochs > 0 and self.current_epoch % self.cfg.evaluate_every_n_epochs == 0
        )
        if evaluate_condition:
            self.generate_and_save_samples(samples_per_class=10)

    def sample(self, n_samples: int, y: int = 0):
        """Algorithm 2: One Step Generation
        x = e + fn(e, t=1, r=0, y)
        # Note: this is slightly modified for t=1 (data) and t=0 (prior)
        """
        x0_shape = (n_samples, self.cfg.num_channels, self.cfg.image_dim, self.cfg.image_dim)
        e = torch.randn(x0_shape).to(self.device) # start with prior
        t = torch.ones(n_samples,).to(self.device) # t = 1
        r = torch.zeros_like(t) # r = 0
        y = torch.ones_like(t) * y # class
        x = e - self.network(e, t=t, r=r, y=y) # actual sampling step
        return x

    def generate_and_save_samples(self, samples_per_class: int = 10):
        """Single Step Sampling using MeanFlow"""
        self.network.eval()
        
        with torch.no_grad():
            samples = torch.cat([
                self.sample(n_samples=samples_per_class, y=class_idx)
                for class_idx in range(self.cfg.num_classes)
            ], dim=0)

            # normalize samples to [0, 1] range
            if self.cfg.normalize:
                samples = (samples - samples.min()) / (samples.max() - samples.min())
            
            # set filename
            filename = f"{self.cfg.save_path}/epoch={self.current_epoch}.png"
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            grid = make_grid(samples, nrow=samples_per_class, padding=2, normalize=False)
            save_image(grid, filename)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: python mean_flow.py <dataset>")
    dataset = sys.argv[1]

    if dataset == "mnist":
        from data import MNISTConfig
        data_config = MNISTConfig()
        patch_size = 7 # 28/7 = 4 patches per row/column (so 16 patches in total)
        num_epochs = 20
        evaluate_every_n_epochs = 5
    elif dataset == "cifar10":
        from data import CIFAR10Config
        data_config = CIFAR10Config()
        patch_size = 8 # 32/8 = 4 patches per row/column (so 16 patches in total)
        num_epochs = 100
        evaluate_every_n_epochs = 10
    else:
        raise ValueError(f"Invalid dataset: {dataset}")

    # setup save path
    save_path = f".data/mean_flow/generated_images/{data_config.dataset}"
    os.makedirs(save_path, exist_ok=True)

    model_cfg = MeanFlowConfig(
        patch_size=patch_size, 
        num_channels=data_config.num_channels,
        image_dim=data_config.image_size,
        num_classes=data_config.num_classes,
        evaluate_every_n_epochs=evaluate_every_n_epochs,  # Generate samples every 5 epochs
        save_path=save_path,
        normalize=True,
    )

    dataloader = create_dataloader(data_config)
    model = MeanFlow(model_cfg)
    callbacks = [ModelCheckpoint(dirpath=f'.data/checkpoints/mean_flow/{data_config.dataset}')]
    trainer = L.Trainer(logger=False, max_epochs=num_epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=dataloader)
