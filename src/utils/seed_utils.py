import random

import numpy as np
import torch


def fix_seed(seed: int) -> None:
    """
    Fixes the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set for random number generation.

    Returns:
        None
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Uncomment the following lines to enforce deterministic algorithms
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = False
