# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""core module"""
from torch import _tensor
from mindspore import jit
from mindspore.common.dtype import *
from mindspore.common.dtype import tensor_type as dtype
from mindspore import Tensor

from . import optim, ops, nn, distributions, cuda, distributed
from .utils import get_default_dtype, set_default_dtype, manual_seed
from .autograd import no_grad, enable_grad, value_and_grad
from .serialization import *
from .ops import *

class device:
    pass

FloatTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor

Size = tuple

long = int64
int = int32
float = float32

def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return Tensor(data, dtype, device=device)
