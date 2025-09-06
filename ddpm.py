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
class DDPMConfig:
    # ddpm args
    T: int = 1000 # Section 4: Experiments
    beta_1: float = 10**-4
    beta_T: float = 0.02

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


class DDPM(L.LightningModule):
    def __init__(self, cfg: DDPMConfig):
        super().__init__()
        self.cfg = cfg

        # set up alpha, beta schedule
        beta_schedule = torch.from_numpy(np.linspace(
            cfg.beta_1, cfg.beta_T, num=cfg.T, endpoint=True, dtype=np.float32
        ))
        # pad with a 0.0 in the beginning to help with indexing {1,...,T}
        beta_schedule = torch.concat([torch.zeros(1), beta_schedule])
        alpha_bar_schedule = torch.cumprod(1 - beta_schedule, dim=0)

        # add to buffer (not model parameters)
        self.register_buffer("beta_schedule", beta_schedule)
        self.register_buffer("alpha_bar_schedule", alpha_bar_schedule)

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

    def get_alpha_t(self, t: Tensor) -> Tensor:
        return 1 - self.beta_schedule[t]
    
    def get_alpha_bar_t(self, t: Tensor) -> Tensor:
        return self.alpha_bar_schedule[t]

    def sigma_t(self, t: Tensor) -> Tensor:
        return self.beta_schedule[t]

    def q_xt_given_x0(self, x0: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Equation 4 in DDPM: Sampling xt from clean sample x0 and noise-scale t"""
        alpha_bar_t = rearrange(self.get_alpha_bar_t(t), "b -> b 1 1 1") # alpha_bar t
        et = torch.randn_like(x0) # e_t term -> N(0, I)
        return torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * et, et

    def training_step(self, batch, batch_idx):
        """"DDPM Algorithm 1: Training"""
        x0, y = batch # image: (b, c, h, w), and label: (b,)
        t = torch.randint(1, self.cfg.T + 1, (x0.shape[0],), device=self.device)
        xt, et = self.q_xt_given_x0(x0, t)
        
        # run DiT to compute et (noise added)
        et_pred = self.network(x=xt, t=t, y=y)

        # compute loss between ut and vt, matching conditional velocity field
        loss = torch.mean((et_pred - et) ** 2)
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
        """"DDPM Algorithm 2: Sampling"""
        xt_shape = (n_samples, self.cfg.num_channels, self.cfg.image_dim, self.cfg.image_dim)
        xt = torch.randn(xt_shape, dtype=torch.float32).to(self.device) # start with noisy sample (t=T)
        t_schedule = np.arange(1, self.cfg.T + 1)[::-1] # reversed
        y = torch.ones(n_samples, device=self.device, dtype=torch.long) * y
        for i in range(self.cfg.T):
            t_i = t_schedule[i]
            z = 0
            if t_i > 1:
                z = torch.randn_like(xt)
            
            t = torch.ones(n_samples, device=self.device, dtype=torch.long) * t_i
            alpha_t = rearrange(1 - self.beta_schedule[t], "b -> b 1 1 1")
            alpha_bar_t = rearrange(self.get_alpha_bar_t(t), "b -> b 1 1 1")
            sigma_t = rearrange(self.sigma_t(t), "b -> b 1 1 1")
            xt = (
                (1/torch.sqrt(alpha_t)) * (
                    xt - self.network(x=xt, t=t,y=y) * (1 - alpha_t)/(torch.sqrt(1 - alpha_bar_t))
                )
                + sigma_t * z
            )
            
        return xt

    def generate_and_save_samples(self, samples_per_class: int = 10):
        """Generate class-conditional samples and save as grid"""
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
        raise ValueError("Usage: python ddpm.py <dataset>")
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
    save_path = f".data/ddpm/generated_images/{data_config.dataset}"
    os.makedirs(save_path, exist_ok=True)

    model_cfg = DDPMConfig(
        patch_size=patch_size, 
        num_channels=data_config.num_channels,
        image_dim=data_config.image_size,
        num_classes=data_config.num_classes,
        evaluate_every_n_epochs=evaluate_every_n_epochs,  # Generate samples every 5 epochs
        save_path=save_path,
        normalize=True,
    )

    dataloader = create_dataloader(data_config)
    model = DDPM(model_cfg)
    callbacks = [ModelCheckpoint(dirpath=f'.data/checkpoints/ddpm/{data_config.dataset}')]
    trainer = L.Trainer(logger=False, max_epochs=num_epochs, callbacks=callbacks)
    trainer.fit(model, train_dataloaders=dataloader)
