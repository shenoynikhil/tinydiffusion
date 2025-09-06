## tinydiffusion
Simple Diffusion Implementations for Educational Purposes. These are restricted to image based data for now.

### Setup
```bash
# install env with uv (lightning fast!)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Models
- [x] Vanilla Flow Matching [[1](https://arxiv.org/abs/2302.00482)],[[2](https://arxiv.org/abs/2210.02747)]
- [x] Denoising Diffusion Probabilistic Models (DDPM) [[1](https://arxiv.org/abs/2006.11239)]
- [ ] Denoising Diffusion Implicit Models (DDIM)
- [ ] Consistency Models


### Acknowledgements
- [DiT](https://github.com/facebookresearch/DiT) from meta for Diffusion transformers.
- [torchCFM](https://github.com/atong01/conditional-flow-matching/tree/main) for neat and simple implementation of flow matching. Scripts on generating images as a grid was taken from this repository.
