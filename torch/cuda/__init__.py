from mindspore import Tensor
from mindspore import get_rng_state, set_rng_state, manual_seed
from mindspore.hal import device_count

FloatTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor

def current_device():
    return -1

def is_available():
    return True
