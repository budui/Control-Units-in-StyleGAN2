from contextlib import contextmanager
from typing import Sequence

import torch
import torch.nn.functional as F
from ignite.handlers import Timer

__all__ = [
    "running_time",
    "empty_cuda_cache",
    "total_chunk",
    "factors_sequence",
    "residual_grid_sample",
    "assert_shape",
    "sequence_chunk"
]


@contextmanager
def running_time(name="this-block", verbose=False):
    t = Timer()
    yield t
    if verbose:
        print(f"{name} consume {t.value()}s")


def empty_cuda_cache() -> None:
    torch.cuda.empty_cache()
    import gc

    gc.collect()


def assert_shape(x: torch.Tensor, shape):
    assert x.ndim == len(shape), f"shape has dim {len(shape)}, but x has dim {x.ndim}"
    for s, es in zip(x.shape, shape):
        if es is None:
            continue
        assert s == es, f"x have shape {x.size()} but the expected shape is {shape}."


def total_chunk(total, chunk_size, drop_last=False):
    seen_amount = 0
    while True:
        if seen_amount >= total:
            break
        cur_chunk_size = min(total - seen_amount, chunk_size)
        if drop_last and cur_chunk_size < chunk_size:
            break
        yield cur_chunk_size
        seen_amount += cur_chunk_size


def sequence_chunk(sequence: Sequence, chunk_size, drop_last=False):
    total = len(sequence)
    i = 0
    for batch in total_chunk(total, chunk_size, drop_last):
        yield sequence[i:batch + i]
        i += batch


def factors_sequence(end_factor, num_factors, start_factor=None, paired_factor=True):
    assert num_factors > 0
    if num_factors == 1:
        return [end_factor, ]
    start_factor = start_factor if start_factor is not None else (-end_factor if paired_factor else 0.0)
    delta = (end_factor - start_factor) / (num_factors - 1)
    return [start_factor + i * delta for i in range(num_factors)]


def residual_grid_sample(image, residual_grid, mode='nearest', padding_mode='reflection', align_corners=False):
    n, c, h, w = image.size()
    assert_shape(residual_grid, (n, 2, h, w))

    device = image.device
    dtype = image.dtype
    grid_h = torch.arange(start=-1, end=1 + 1e-10, step=2 / (h - 1), device=device, dtype=dtype)
    grid_w = torch.arange(start=-1, end=1 + 1e-10, step=2 / (w - 1), device=device, dtype=dtype)
    zero_grid = torch.stack(torch.meshgrid(grid_h, grid_w)[::-1])
    grid = (residual_grid + zero_grid.unsqueeze(dim=0)).permute(0, 2, 3, 1)
    return F.grid_sample(image, grid, mode, padding_mode, align_corners)
