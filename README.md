# Patchwork
The implementation of preprint Patchwork: A compact representation for 3D polygonal shapes. Visual results are available at our [project page](https://ankbzpx.github.io/patchwork-page/).

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