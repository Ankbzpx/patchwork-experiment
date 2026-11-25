import os

import numpy as np
import pandas as pd

from icecream import ic


if __name__ == "__main__":
    dataset_list = ["abc", "thingi10k", "roof_modeling", "shapenet"]
    method_list = ["ours_explicit"]

    metrics = ["chamfer", "hausdorff", "f1", "normal", "iou"]
    multiplier = [1e3, 1e2, 1e2, 1e2, 1e2]
    metrics_column = [
        "item",
        "chamfer",
        "hausdorff",
        "f1",
        "normal",
        "iou",
    ]

    os.makedirs("metrics/collect", exist_ok=True)

    def append_collection(tag, data, collection):
        metric_collect = [tag]
        for metric, mul in zip(metrics, multiplier):
            metric_collect.append(mul * data[metric].mean())

        metrics_frame = pd.DataFrame([metric_collect], columns=metrics_column)

        if collection is None:
            collection = metrics_frame
        else:
            collection = pd.concat([collection, metrics_frame])

        return collection

    for dataset in dataset_list:
        collection = None
        for method in method_list:
            tag = method
            csv_path = os.path.join("metrics", f"{tag}_{dataset}.csv")
            if os.path.exists(csv_path):
                data = pd.read_csv(csv_path)
                collection = append_collection(tag, data, collection)

        collection.to_csv(os.path.join("metrics", "collect", f"{dataset}.csv"))
