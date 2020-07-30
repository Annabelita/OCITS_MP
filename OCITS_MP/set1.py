import torch
import gc
import numpy as np
import timeit
import models
from training import simple_train
import itertools
from tqdm import tqdm
import sys
import pandas as pd


def main():
    resnet_types = [18, 34, 50]
    training_styles = ["1GPU", "2GPU_sync", "2GPU_async"]
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    results = []
    for idx, (resnet_type, training_style, batch_size) in enumerate(
        tqdm(list(itertools.product(resnet_types, training_styles, batch_sizes)))
    ):
        if idx >= 14:
            continue

        if training_style == "1GPU":
            split_size = None
        else:
            split_size = 20

        res = run(resnet_type, training_style, batch_size, split_size)

        print(res)

        res = {
            "resnet_type": resnet_type,
            "training_style": training_style,
            "batch_size": batch_size,
            "split_size": split_size,
            "time_mean": res[0],
            "time_std": res[1],
            "memory_1": np.nan if res[2] is np.nan else res[2][0],
            "memory_2": np.nan if res[2] is np.nan else res[2][1],
            "memory_3": np.nan if res[2] is np.nan else res[2][2],
        }
        print(res)

        results.append(res)

        pd.DataFrame(results).to_pickle("set1_results2.pkl")


def run(resnet_type, training_style, batch_size, split_size, num_repeat=10):

    if training_style == "1GPU":
        assert split_size is None
        model_string = f"models.resnet{resnet_type}(num_classes=2).to('cuda:1')"
        model = eval(model_string)
    elif training_style == "2GPU_sync":
        model_string = f"models.ResNet_2GPU_Sync(resnet_type={resnet_type})"
        model = models.ResNet_2GPU_Sync(resnet_type=resnet_type)
    elif training_style == "2GPU_async":
        model_string = f"models.ResNet_2GPU_Async(resnet_type={resnet_type}, split_size={split_size})"
        model = models.ResNet_2GPU_Async(resnet_type=resnet_type, split_size=split_size)
    elif training_style == "3GPU_async":
        assert resnet_type == 100
        model_string = f"models.ResNet_3GPU_Async(split_size={split_size})"
        model = models.ResNet_3GPU_Async(split_size=split_size)
    else:
        raise ValueError(f"Trainin style {training_style} is unknown!")

    try:
        memory = simple_train(model, batch_size)

        setup = f"model = {model_string}"
        stmt = f"simple_train(model, {batch_size})"
        run_times = timeit.repeat(
            stmt, setup, number=1, repeat=num_repeat, globals=globals()
        )

        mean, std = np.mean(run_times), np.std(run_times)
    except RuntimeError:
        gc.collect()
        torch.cuda.empty_cache()

        mean = std = memory = np.nan

    return mean, std, memory


if __name__ == "__main__":
    main()
