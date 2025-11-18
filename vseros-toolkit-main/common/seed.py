import importlib
import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    spec = importlib.util.find_spec("torch")
    if spec is not None:
        torch = importlib.import_module("torch")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


__all__ = ["set_global_seed"]
