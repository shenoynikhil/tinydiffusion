## tinydiffusion
Simple implementations of Diffusion/Flow-Matching styled models. This is mostly for educational purposes. The implementations are not optimized for performance but rather for readability. 

Please open an issue if you have suggestions for improvements!

### Setup
```bash
# install env with uv (lightning fast!)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
```

### Generative Models
- [x] Conditional Flow Matching [[1](https://arxiv.org/abs/2302.00482)][[2](https://arxiv.org/abs/2210.02747)]:
```bash
python flow_matching.py <cifar10|mnist>
```

- [x] Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) [[1](https://arxiv.org/abs/2006.11239)][[2](https://arxiv.org/abs/2010.02502)]:
```bash
python ddpm.py <mnist|cifar10> [--sampling_type {ddpm,ddim}] [--ddim_steps 50] [--ddim_stochasticity] [--T 1000]
```

- [x] Mean Flows for One-Step Generative Modelling [[1](https://arxiv.org/abs/2505.13447)]:
```bash
python mean_flow.py <mnist|cifar10>
```

### WIP
- [ ] Consistency Models
- [ ] Inductive Moment Matching


### Acknowledgements
- [DiT](https://github.com/facebookresearch/DiT) from meta for Diffusion transformers.
