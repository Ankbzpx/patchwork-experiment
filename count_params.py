from glob import glob
import os
import json
import numpy as np

from icecream import ic


if __name__ == "__main__":
    for base_path in [
        "exp_N_1024/roof_modeling",
        "exp_N_16384/abc",
        "exp_N_16384/thingi10k",
        "exp_N_65536/shapenet",
        "exp_N_1048576/scale",
    ]:
        json_list = (
            glob(os.path.join(base_path, "**/**", "meta.json"))
            if base_path.endswith("shapenet")
            else glob(os.path.join(base_path, "**", "meta.json"))
        )

        param_total = 0
        params = []
        num_shape = len(json_list)
        for j_path in json_list:
            with open(j_path) as f:
                n_param = json.load(f)["n_params"]
                param_total += n_param
                params.append(n_param)

        params = np.array(params)
        # print(4 * np.max(params), 4 * np.min(params))
        print(4 * param_total / num_shape)
