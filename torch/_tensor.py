import copy
import mindspore
from mindspore import Tensor, ops
from mindspore.common._stub_tensor import StubTensor

def numel(self):
    return ops.size(self)

Tensor.numel = numel
setattr(StubTensor, 'numel', numel)

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
