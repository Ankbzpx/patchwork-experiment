import argparse
from glob import glob
import os

from icecream import ic


if __name__ == "__main__":
    base_folder = "data/polygon"
    exp_dir = "exp_polygon"

    pc_list = glob(f"{base_folder}/*.npz")
    for pc_path in pc_list:
        model_name = pc_path.removeprefix(base_folder)[1:-4]
        print(f"{model_name}")
        save_path = f"{exp_dir}/result_meshes/{model_name}.obj"
        if os.path.exists(save_path):
            continue
        cmd = f"python fit_geo_2d.py --model_name {model_name} --exp_dir {exp_dir}"
        os.system(cmd)
