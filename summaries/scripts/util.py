import logging
import numpy as np
import os
import torch as th


def setup():
    """
    General script setup based on environment variables.
    """
    level = os.environ.get('LOGLEVEL', 'warning')
    logging.basicConfig(level=level.upper())

    seed = os.environ.get('SEED')
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)
        th.manual_seed(seed)
