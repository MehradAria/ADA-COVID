"""Reproducibility and environment utilities."""

import os
import random

import numpy as np
import tensorflow as tf

from .config import SEED


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)


def print_environment_info() -> None:
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
    print(f"Eager execution: {tf.executing_eagerly()}")
