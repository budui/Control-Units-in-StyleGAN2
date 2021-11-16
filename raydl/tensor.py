import numbers
from typing import Iterable, Union, Optional, Tuple

import cv2
import numpy as np
import torch

__all__ = ["grid_transpose", "create_heatmap", "constant", "is_scalar", "nan_to_num"]

_constant_cache = dict()


def is_scalar(v):
    if isinstance(v, numbers.Number):
        return True
    if torch.is_tensor(v) and torch.numel(v) == 1:
        return True
    return False


try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:
    def nan_to_num(input, nan=0.0, posinf=None, neginf=None, *, out=None):  # pylint: disable=redefined-builtin
        # Replace NaN/Inf with specified numerical values.
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out)


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    """
    Cached construction of constant tensors. Avoids CPU=>GPU copy when the
    same constant is used multiple times.
    :param value:
    :param shape:
    :param dtype:
    :param device:
    :param memory_format:
    :return:
    """
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device('cpu')
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype, device, memory_format)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


def create_heatmap(
        images: torch.Tensor,
        range_min: Optional[float] = None,
        range_max: Optional[float] = None,
        scale_each: bool = False,
        color_map: str = "jet",
        return_tensor: bool = True
) -> Union[torch.Tensor, Tuple[np.array, ...]]:
    """
    create heatmap from BxHxW tensor.
    :param images: Tensor[BxHxW]
    :param range_min: max value used to normalize the image. By default, min and max are computed from the tensor.
    :param range_max: min value used to normalize the image. By default, min and max are computed from the tensor.
    :param scale_each: If True, scale each image in the batch of images separately rather
     than the (min, max) over all images. Default: False.
    :param color_map: The colormap to apply, colormap from
     https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html#ga9a805d8262bcbe273f16be9ea2055a65
    :param return_tensor: if True, return Tensor[Bx3xHxW], otherwise return tuple of numpy.array(0-255)
    :return:
    """
    device = images.device
    assert images.dim() == 3
    try:
        color_map = getattr(cv2, f"COLORMAP_{color_map.upper()}")
    except AttributeError:
        raise ValueError(f"invalid color_map {color_map}")

    with torch.no_grad():
        images = images.detach().clone().to(dtype=torch.float32, device=torch.device("cpu"))

        if range_min is None:
            range_min = images.amin(dim=[-1, -2], keepdim=True) if scale_each else images.amin()
        if range_max is None:
            range_max = images.amax(dim=[-1, -2], keepdim=True) if scale_each else images.amax()

        heatmaps = []
        for m in images.add_(-range_min).div_(range_max - range_min + 1e-5).clip_(0.0, 1.0):
            heatmaps.append(cv2.applyColorMap(np.uint8(m.numpy() * 255), color_map))

        if return_tensor:
            heatmaps = torch.from_numpy(np.stack(heatmaps)).permute(0, 3, 1, 2)
            # BGR -> RGB & [0, 255] -> [-1, 1]
            heatmaps = (heatmaps[:, [2, 1, 0], :, :].contiguous().float().to(device) / 255 - 0.5) * 2
            return heatmaps
        return tuple(heatmaps)


def grid_transpose(tensors: Union[torch.Tensor, Iterable], original_nrow: Optional[int] = None) -> torch.Tensor:
    """
    batch tensors transpose.
    :param tensors: Tensor[(ROW*COL)*D1*...], or Iterable of same size tensors.
    :param original_nrow: original ROW
    :return: Tensor[(COL*ROW)*D1*...]
    """
    assert torch.is_tensor(tensors) or isinstance(tensors, Iterable)
    if not torch.is_tensor(tensors) and isinstance(tensors, Iterable):
        seen_size = None
        grid = []
        for tensor in tensors:
            if seen_size is None:
                seen_size = tensor.size()
                original_nrow = original_nrow or len(tensor)
            elif tensor.size() != seen_size:
                raise ValueError("expect all tensor in images have the same size.")
            grid.append(tensor)
        tensors = torch.cat(grid)

    assert original_nrow is not None

    cell_size = tensors.size()[1:]

    tensors = tensors.reshape(-1, original_nrow, *cell_size)
    tensors = torch.transpose(tensors, 0, 1)
    return torch.reshape(tensors, (-1, *cell_size))
