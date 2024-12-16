import copy
import mindspore
from mindspore import Tensor, ops
from mindspore.common._stub_tensor import StubTensor
from mindspore._c_expression import Tensor as Tensor_
from ._utils import _rebuild_tensor_v2

def to_dense(self):
    return self

Tensor.to_dense = to_dense
StubTensor.to_dense = to_dense

Tensor._base = None
StubTensor._base = None

@property
def data(self):
    return self

@data.setter
def data(self, new_value):
    self.assign_value(new_value)

Tensor.data = data
StubTensor.data = data

def numel(self):
    return ops.size(self)

Tensor.numel = numel
setattr(StubTensor, 'numel', numel)
Tensor.nelement = numel
StubTensor.nelement = numel

StubTensor.__hash__ = Tensor.__hash__

def _repeat(self, *sizes):
    return ops.tile(self, tuple(sizes))

Tensor.repeat = _repeat
StubTensor.repeat = _repeat

def no_action(self):
    return self

Tensor.cuda = no_action
StubTensor.cuda = no_action
Tensor.cpu = no_action
StubTensor.cpu = no_action


def size(self, dim=None):
    if dim is None:
        return self.shape
    assert isinstance(dim, int), f'`dim` must be int but got {type(dim)}'
    return self.shape[dim]

Tensor.size = size
StubTensor.size = size

def dim(self):
    return self.ndim

Tensor.dim = dim
StubTensor.dim = dim

def clone(self):
    return copy.deepcopy(self)

Tensor.clone = clone
StubTensor.clone = clone

def __or__(self, other):
    if isinstance(other, (int, bool, float, Tensor)):
        return ops.bitwise_or(self.to(mindspore.int32), other.to(mindspore.int32)).bool()
    raise TypeError("Unsupported operand type(s) for |: 'Tensor' and '{}'".format(type(other)))

Tensor.__or__ = __or__
StubTensor.__or__ = __or__

Tensor.device = 'NO_INFO'
StubTensor.device = 'NO_INFO'

def sum(self, dim=-1):
    return ops.sum(self, dim)

Tensor.sum = sum
StubTensor.sum = sum

def div_(self, value, *, rounding_mode=None):
    out = self.div(value, rounding_mode=rounding_mode)
    self.assign_value(out)

Tensor.div_ = div_
StubTensor.div_ = div_

def __reduce_ex__(self, protocol):
    if isinstance(self, StubTensor):
        data = Tensor_(self.stub_sync())
    else:
        data = Tensor_(self)
    storage_offset = 0
    size = data._shape
    stride = data.stride()
    requires_grad = False
    args = (data, storage_offset, size, stride, requires_grad, None, None)
    return (
        _rebuild_from_type_v2, (_rebuild_tensor_v2, type(self), args, None))


Tensor.__reduce_ex__ = __reduce_ex__
StubTensor.__reduce_ex__ = __reduce_ex__

def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    return ret

def detach(self):
    return self

Tensor.detach = detach
StubTensor.detach = detach
