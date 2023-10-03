import os
import random
import numpy as np
import torch


RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
