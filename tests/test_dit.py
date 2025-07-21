import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from dit import DiT

def test_dit():

    model = DiT(patch_size=7, num_channels=3, image_dim=28, hidden_dim=16)
    x = torch.randn(2, 3, 28, 28) # 2 images of size 1 x 28 x 28
    c = torch.randn(2, 1)

    assert model(x, c).shape == (2, 3, 28, 28)
