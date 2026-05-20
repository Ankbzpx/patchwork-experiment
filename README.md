# Patchwork
The implementation of preprint [Patchwork: A compact representation for 3D polygonal shapes](https://arxiv.org/abs/2605.16266). Visual results are available at our [project page](https://ankbzpx.github.io/patchwork-page/).

## Environment Setup
```
conda create -n patchwork python=3.10 -y
conda activate patchwork
pip install -r requirements.txt
```

## Playground
```bash
python playground.py
```

## Fitting
Download the `data.zip` from release page and unzip it to the project root folder.

### JAX
```
python fit_geo_2d.py --model_name wost --exp_dir exp --vis

python fit_geo.py --model_name smooth/bunny --exp_dir exp --res 512 --vis
```

### Pytorch
```
python fit_geo_pytorch.py --model_name smooth/bunny --exp_dir exp --res 512 --vis
```

## Side notes
1. The PyTorch version supports accurate second-order derivatives via [FlexAttention](https://docs.pytorch.org/docs/2.12/nn.attention.flex_attention.html). Alternatively, it can be approximated with pass through (`--grad_pass_through` flag) for half the computational cost (i.e., ~5 mins on RTX 3090).
2. `playground.py` reproduces the illustrative Fig.3 in the preprint, showcasing the close-to-optimal patchworks. They are initialized from our custom ADMM solver (`admm.py`), which does not scale and is out of the scope of the main method. We include it here in case it is useful to the community.