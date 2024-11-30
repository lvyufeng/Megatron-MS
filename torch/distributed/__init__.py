from mindspore.communication import GlobalComm, get_group_size as get_world_size, get_rank, create_group
from mindspore.mint.distributed import init_process_group, destroy_process_group
from mindspore.communication.comm_func import barrier, all_reduce as all_reduce_, broadcast as broadcast_
from mindspore.communication._comm_helper import _ExistingGroup, _HCCL_TEST_AVAILABLE
from mindspore.ops import ReduceOp, assign

import torch

_HCCL_TEST_AVAILABLE = True

ProcessGroup = str

def is_initialized():
    return GlobalComm.INITED

_group_count = 0
NON_GROUP_MEMBER = -100

def _process_group_name(ranks):
    global _group_count
    pg_name = str(_group_count)
    _group_count += 1
    return pg_name

def new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None):
    pg_name = _process_group_name(ranks)
    ranks = list(ranks)
    if pg_name in _ExistingGroup.ITEMS:
        raise ValueError(f'{pg_name} already exist.')
    if get_rank() not in ranks:
        return NON_GROUP_MEMBER
    create_group(pg_name, ranks)
    return pg_name

class Backend:
    class Options:
        def __init__(self, backend: str, timeout = ...) -> None: ...
        @property
        def backend(self) -> str: ...
        @property
        def _timeout(self) -> int: ...
        @_timeout.setter
        def _timeout(self, val) -> None: ...

    def __init__(
        self,
        rank: int,
        size: int,
    ) -> None: ...
    @property
    def supports_splitting(self) -> bool: ...
    @property
    def options(self) -> Options: ...
    def rank(self) -> int: ...
    def size(self) -> int: ...
    def eager_connect_single_device(self, device: None) -> None: ...
    def _set_sequence_number_for_group(self) -> None: ...
    def _set_default_timeout(self, timeout) -> None: ...


class ProcessGroupNCCL(Backend):
    class NCCLConfig:
        blocking: int
        cga_cluster_size: int
        min_ctas: int
        max_ctas: int

    class Options(Backend.Options):
        config: dict
        is_high_priority_stream: bool
        split_from: dict
        split_color: int
        global_ranks_in_group: list[int]
        group_name: str

        def __init__(self, is_high_priority_stream: bool = False): ...

    def __init__(
        self,
        store,
        rank: int,
        size: int,
        options: Options,
    ) -> None: ...
    def _group_start(self) -> None: ...
    def _group_end(self) -> None: ...
    def _set_default_timeout(self, timeout) -> None: ...
    def _shutdown(self) -> None: ...
    def perform_nocolor_split(self, device) -> None: ...
    def register_mem_pool(self, pool) -> None: ...
    def deregister_mem_pool(self, pool) -> None: ...
    def comm_split_count(self) -> int: ...
    def _add_ephemeral_timeout(self, timeout) -> None: ...
    def abort(self) -> None: ...
    def _is_initialized(self) -> bool: ...
    @property
    def uid(self) -> int: ...
    @property
    def options(self) -> Options: ...  # type: ignore[override]

def all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP, async_op=False):
    if tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
    new_tensor, handle = all_reduce_(tensor, op, group, async_op)
    return new_tensor, handle

def broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP):
    if tensor.dtype == torch.int64:
        tensor = tensor.to(torch.int32)
    new_tensor = broadcast_(tensor, src, group)
    return new_tensor
