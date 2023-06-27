import os
import random

import numpy as np
import torch


def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

def unnormalize(data, mean, std):
    return data * std + mean

def seed_rngs(seed: int, pytorch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if pytorch:
        torch.manual_seed(seed)


def set_cudnn(deterministic: bool = False, benchmark: bool = True) -> None:
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark