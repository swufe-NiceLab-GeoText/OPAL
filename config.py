import random
from typing import Dict

import numpy as np
import torch


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


TRAIN_CONFIG: Dict[str, float | int] = {
    "epochs": 80,
    "batch_size": 512,
    "learning_rate": 1e-3,
    "hidden_dim": 256,
    "representation_dim": 200,
    "dropout_rate": 0.2,
    "num_attention_heads": 4,
    "warmup_epochs": 10,
    "weight_decay": 1e-4,
}


DTCOIF_CONFIG: Dict[str, float] = {
    "alpha": 1.0,
    "lambda_eif": 0.9,
    "delta": 0.15,
    "lambda_orth": 1.0,
    "s_max": 5.0,
    "eps_g": 1e-3,
    "eps": 1e-8,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


