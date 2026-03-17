import argparse
from glob import glob
import os

from icecream import ic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder")
    parser.add_argument("--N", type=int, default=16384, help="Num samples for init")
    args = parser.parse_args()

    folder = args.folder
    base_folder = "data/pc"

    w_mse = 1.0
    w_normal = 1.0
    w_reg = 1.0
    N = args.N

    exp_dir = f"exp_geo_{N}_{w_mse}_{w_normal}_{w_reg}"

    pc_list = glob(f"{base_folder}/{folder}/*.ply")
    for pc_path in pc_list:
        model_name = pc_path.removeprefix(base_folder)[1:-4]
        print(f"{model_name}")
        save_path = f"{exp_dir}/result_meshes/{model_name}.obj"
        if os.path.exists(save_path):
            continue
        cmd = f"python fit_geo.py --model_name {model_name} --exp_dir {exp_dir} --N {N} --w_mse {w_mse} --w_normal {w_normal} --w_reg {w_reg} --res 512"
        os.system(cmd)
