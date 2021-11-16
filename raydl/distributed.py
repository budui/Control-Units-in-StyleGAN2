import contextlib

import torch


@contextlib.contextmanager
def ddp_sync(module: torch.nn.Module, sync: bool):
    """
    allow DDP sync or not.
    :param module: if module is not DDP-wrapped module, do nothing.
    :param sync: enable sync between process or not if module is DDP-wrapped module
    :return:
    """
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield
