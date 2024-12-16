"""utils"""
import random
import numpy as np
import mindspore
from ..configs import DEFAULT_DTYPE, GENERATOR_SEED

import mindspore as ms

def set_default_dtype(dtype):
    """set default dtype"""
    global DEFAULT_DTYPE
    DEFAULT_DTYPE = dtype

def get_default_dtype():
    """get default dtype"""
    return DEFAULT_DTYPE

def manual_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)
    if GENERATOR_SEED:
        mindspore.manual_seed(seed)

def use_deterministic_algorithms(flag: bool):

    ms.set_context(deterministic='ON' if flag else 'OFF')