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
import os
import platform
from packaging import version
import mindspore
from mindspore import context
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error


if 'RANK_TABLE_FILE' in os.environ:
    del os.environ['RANK_TABLE_FILE']
DEVICE_TARGET = os.environ.get('DEVICE_TARGET', None)

if DEVICE_TARGET is not None and DEVICE_TARGET in ('CPU', 'GPU', 'Ascend'):
    context.set_context(device_target=DEVICE_TARGET)

if platform.system().lower() == 'linux':
    SOC = MSContext.get_instance().get_ascend_soc_version()
    if ('910b' not in SOC and '310' not in SOC) or version.parse(mindspore.__version__) < version.parse('2.4.0'):
        os.environ["MS_ALLOC_CONF"] = 'enable_vmm:True,vmm_align_size:2MB'

    if SOC in ('ascend910', 'ascend310b'):
        context.set_context(ascend_config={"precision_mode": "allow_mix_precision"})

from torch import _tensor
from mindspore import jit
from mindspore.common.dtype import *
from mindspore.common.dtype import tensor_type as dtype
from mindspore import Tensor, default_generator, Generator
from mindspore.hal import Stream
from mindspore import multiprocessing

inf = float("inf")
nan = float("nan")

class device:
    def __init__(self, name):
        pass

from torch import amp as amp
from . import optim, ops, nn, distributions, cuda, distributed
from .utils import get_default_dtype, set_default_dtype, manual_seed
from .autograd import no_grad, enable_grad, value_and_grad
from .serialization import *
from .ops import *


FloatTensor = Tensor
HalfTensor = Tensor
BFloat16Tensor = Tensor
LongTensor = Tensor

Size = tuple

long = int64
int = int32
float = float32
bool = bool_

def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
    return Tensor(data, dtype)

strided = None
contiguous_format = None
preserve_format = None

AUTO_CAST_DTYE = {
    'cuda': float16,
    'cpu': bfloat16,
    'npu': float16
}

def set_autocast_dtype(device_type, dtype):
    assert device_type in AUTO_CAST_DTYE.keys(), f'{device_type} is not in {AUTO_CAST_DTYE.keys()}'
    AUTO_CAST_DTYE[device_type] = dtype

def get_autocast_dtype(device_type):
    return AUTO_CAST_DTYE[device_type]

def is_autocast_enabled(device_type):
    return device_type in AUTO_CAST_DTYE.keys()
