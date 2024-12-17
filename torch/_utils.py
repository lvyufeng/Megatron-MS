import mindspore
from mindspore import Tensor
from .ops import empty, narrow


element_size_map = {
    mindspore.float16: 2,
    mindspore.float32: 3,
    mindspore.bfloat16: 2,
    mindspore.int64: 4,
    mindspore.uint8: 1,
    mindspore.int8: 1,
    mindspore.bool_: 1
}

def _element_size(dtype):
    return element_size_map[dtype]

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return Tensor._flatten_tensors(tensors)


def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        if numel == 0:
            outputs.append(empty(0, flat.dtype))
        else:
            outputs.append(narrow(flat, 0, offset, numel).view(tensor.shape))
            offset += numel
    return outputs

def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    pass

def is_version_ge(current_version, base_version):
    version_split_char = '.'
    if version_split_char not in base_version or version_split_char not in current_version:
        raise ValueError("The version string will conain the '.'."
                         "For example, current_version 1.8.1, base_version 1.11.0.")
    for x, y in zip(current_version.split(version_split_char), base_version.split(version_split_char)):
        if not x.isdigit() or not y.isdigit():
            continue
        if int(x) != int(y):
            return int(x) >= int(y)
    return True

def check_valid_version(valid_version):
    return is_version_ge(mindspore.__version__, valid_version)
