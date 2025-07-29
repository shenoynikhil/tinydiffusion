from dataclasses import dataclass
import os
import sys

import numpy as np
import torch
from torch import Tensor
import lightning as L
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint

from dit import DiT
from data import create_dataloader
from torchvision.utils import make_grid, save_image


@dataclass
class FlowMatchingConfig:
    # flow matching arg
    prior_sigma: float = 1.0
    eps: float = 1.e-4 # for numerical stability
    interpolation_type: str = "linear"
    integration_type: str= "euler"

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


class VanillaFlowMatching(L.LightningModule):
    def __init__(self, cfg: FlowMatchingConfig):
        super().__init__()
        self.cfg = cfg

        self.network = DiT(
            input_size=self.cfg.image_dim,
            patch_size=self.cfg.patch_size,
            in_channels=self.cfg.num_channels,
            depth=self.cfg.depth,
            num_heads=self.cfg.num_heads,
            hidden_size=self.cfg.hidden_size,
            mlp_ratio=self.cfg.mlp_ratio,
            num_classes=self.cfg.num_classes,
        )

    def get_xt_and_ut(self, x1: Tensor, x0: Tensor, t: Tensor):
        if self.cfg.interpolation_type == "linear":
            t_reshaped = rearrange(t, "b -> b 1 1 1")
            xt = t_reshaped * x1 + (1 - t_reshaped) * x0 # unsqueeze out t to make sure, its 4D
            ut = x1 - x0 # ground truth velocity field, based on linear interpolation
        else:
            raise NotImplementedError

        return xt, ut

    def training_step(self, batch, batch_idx):
        x1, y = batch # image: (b, c, h, w), and label: (b,)
        x0 = torch.randn_like(x1) * self.cfg.prior_sigma # gaussian prior
        t = torch.clip(torch.rand(x1.shape[0],), min=self.cfg.eps, max=1 - self.cfg.eps).to(
            self.device
        )
        
        # interpolate and compute conditional velocity field
        xt, ut = self.get_xt_and_ut(x1=x1, x0=x0, t=t)

        # run DiT to compute predicted velocity field
        vt = self.network(x=xt, t=t, y=y)

        # compute loss between ut and vt, matching conditional velocity field
        loss = torch.mean((ut - vt) ** 2)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.cfg.lr)

    def on_train_epoch_end(self):
        evaluate_condition = (
            self.cfg.evaluate_every_n_epochs > 0 and self.current_epoch % self.cfg.evaluate_every_n_epochs == 0
        )
        if evaluate_condition:
            self.generate_and_save_samples(samples_per_class=10, n_steps=50)

    def sample(self, n_samples: int, y: int = 0, n_steps: int = 10):
        x0_shape = (n_samples, self.cfg.num_channels, self.cfg.image_dim, self.cfg.image_dim)
        x = torch.randn(x0_shape).to(self.device) # start with prior
        t_schedule = np.linspace(self.cfg.eps, 1., n_steps + 1)
        for i in range(n_steps):
            t_delta = t_schedule[i + 1] - t_schedule[i]
            t = torch.ones(n_samples, device=self.device) * t_schedule[i]
            y = torch.ones(n_samples, device=self.device, dtype=torch.long) * y
            x = x + t_delta * self.network(x=x, t=t, y=y)
        return x

    def generate_and_save_samples(self, samples_per_class: int = 10, n_steps: int = 50):
        """Generate class-conditional samples and save as grid"""
        self.network.eval()
        
        with torch.no_grad():
            samples = torch.cat([
                self.sample(n_samples=samples_per_class, y=class_idx, n_steps=n_steps)
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
        raise ValueError("Usage: python flow_matching.py <dataset>")
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
    save_path = f".data/flow_matching/generated_images/{data_config.dataset}"
    os.makedirs(save_path, exist_ok=True)

    model_cfg = FlowMatchingConfig(
        patch_size=patch_size, 
        num_channels=data_config.num_channels,
        image_dim=data_config.image_size,
        num_classes=data_config.num_classes,
        evaluate_every_n_epochs=evaluate_every_n_epochs,  # Generate samples every 5 epochs
        save_path=save_path,
        normalize=True,
    )

    dataloader = create_dataloader(data_config)
    model = VanillaFlowMatching(model_cfg)
    callbacks = [ModelCheckpoint(dirpath=f'.data/checkpoints/flow_matching/{data_config.dataset}')]
    trainer = L.Trainer(logger=False, max_epochs=num_epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=dataloader)
