import math
import os
import pickle
import warnings
from io import BytesIO
from pathlib import Path
from typing import Union, Optional, List, Tuple, Text, Iterable, Sequence

import lmdb
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets.folder import is_image_file, default_loader
from torchvision.utils import make_grid

__all__ = [
    "load_images",
    "save_images",
    "draw_captions_over_image",
    "resize_images",
    "images_files",
    "pil_loader",
    "LMDBCacheLoader"
]


def resize_images(images: torch.Tensor, resize=None, resize_mode: str = "bilinear") -> torch.Tensor:
    """
    resize images, when resize is not None.
    :param images: torch.Tensor[NxCxHxW]
    :param resize: None means do nothing, or target_size[int]. target_size will be convert to (target_size, target_size)
    :param resize_mode: interpolate mode.
    :return: resized images.
    """
    if resize is None:
        return images
    if isinstance(resize, (int, float)):
        resize = (int(resize), int(resize * images.shape[-1] / images.shape[-2]))
    resize = (resize, resize) if isinstance(resize, int) else resize
    if resize[0] != images[0].size(-2) or resize[1] != images[0].size(-1):
        align_corners = False if resize_mode in ["linear", "bilinear", "bicubic", "trilinear"] else None
        images = F.interpolate(images, size=resize, mode=resize_mode, align_corners=align_corners)
    return images


def pil_loader(path: str, mode="RGB") -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert(mode)


def load_images(
        images_path: Union[str, Path, Iterable],
        resize: Optional[Union[int, Tuple]] = None,
        value_range: Tuple[int, int] = (-1, 1),
        device: Union[torch.device, str] = torch.device("cuda"),
        image_mode: str = "RGB",
        resize_mode: str = "bilinear",
) -> torch.Tensor:
    """
    read images into tensor
    :param images_path:
    :param resize:
    :param value_range:
    :param device:
    :param image_mode: accept a string to specify the mode of image, must in
     https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
    :param resize_mode: accept a string to specify the mode of resize(interpolate), must in
     https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
    :return: images (Tensor[num_images, image_channels, image_height, image_width])
    """
    if isinstance(images_path, (str, Path)):
        images_path = [images_path, ]
    images = []
    for image_path in images_path:
        pil_image = pil_loader(image_path, image_mode)
        # 1xCxHxW, value_range: [0, 1]
        image = torchvision.transforms.functional.to_tensor(pil_image).unsqueeze_(0)
        images.append(resize_images(image, resize=resize, resize_mode=resize_mode))
    images = torch.cat(images).to(device)
    images = images * (value_range[1] - value_range[0]) + value_range[0]
    return images


def draw_captions_over_image(
        pil_image: Image,
        captions: Sequence,
        grid_cell_size: Tuple[int, int],
        grid_cell_padding: int,
        caption_color: str = "#ff0000",
        caption_font="DejaVuSans.ttf"
) -> Image:
    """
    draw captions over grid image. use grid_cell_size to specify minimal cell size in grid.
    :param pil_image: grid image
    :param captions: a sequence of value in (None, str, tuple).
     value can be None, which means skip this grid cell;
     can be str, which is captions;
     can be tuple of (caption, color), for specify color for this caption.
    :param grid_cell_size: tuple (height, width)
    :param grid_cell_padding: padding when make grid
    :param caption_color: the color of the captions, default is red "#ff0000"
    :param caption_font: the font of the captions, default is DejaVuSans.ttf,
        will find font as https://pillow.readthedocs.io/en/latest/reference/ImageFont.html#PIL.ImageFont.truetype
    :return:
    """
    h, w = grid_cell_size
    padding = grid_cell_padding
    nrow = pil_image.width // w
    im_draw = ImageDraw.Draw(pil_image)
    try:
        im_font = ImageFont.truetype(caption_font, size=max(h // 10, 12))
    except OSError:
        warnings.warn(f"can not find {caption_font}, so use the default font, better than nothing")
        im_font = ImageFont.load_default()

    for i, cap in enumerate(captions):
        if cap is None:
            continue
        cap, fill_color = cap, caption_color if not isinstance(cap, (tuple, list)) else cap
        im_draw.text(
            ((i % nrow) * (w + padding) - padding, (i // nrow) * (h + padding) - padding),
            cap,
            fill=fill_color,
            font=im_font
        )
    return pil_image


def infer_pleasant_nrow(length: int):
    sqrt_nrow_candidate = int(math.sqrt(length))
    if sqrt_nrow_candidate ** 2 == length:
        return sqrt_nrow_candidate
    return 2 ** int(math.log2(math.sqrt(length)) + 1)


def save_images(
        images: Union[torch.Tensor, List[torch.Tensor]],
        save_path: Union[Text, Path, Sequence],
        captions: Optional[Union[bool, Sequence]] = None,
        resize: Optional[Union[int, Tuple[int, int]]] = None,
        separately: bool = False,
        nrow: Optional[int] = None,
        normalize: bool = True,
        value_range: Optional[Tuple[int, int]] = (-1, 1),
        scale_each: bool = False,
        padding: int = 0,
        pad_value: int = 0,
        caption_color: str = "#ff0000",
        caption_font="DejaVuSans.ttf"
):
    """
    save images
    :param images: (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
    :param save_path: path to save images
    :param captions:
    :param resize: if not None, resize images.
    :param separately: if True, save images separately rather make grid
    :param nrow: Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``.
    :param normalize: If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
    :param value_range: tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image.
    :param scale_each: If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
    :param padding: amount of padding. Default: ``0``.
    :param pad_value:  Value for the padded pixels. Default: ``0``.
    :param caption_color: the color of the captions, default is red "#ff0000"
    :param caption_font: the font of the captions, default is DejaVuSans.ttf,
        will find font as https://pillow.readthedocs.io/en/latest/reference/ImageFont.html#PIL.ImageFont.truetype
    :return: None
    """
    assert not (isinstance(save_path, (list, tuple)) and not separately), \
        f"{save_path} separately: {separately}"
    if isinstance(captions, bool) and captions:
        captions = list(map(str, range(len(images))))
    if not torch.is_tensor(images):
        images = torch.cat(images)
    images = resize_images(images, resize=resize)

    if not separately:
        if nrow is None:
            nrow = infer_pleasant_nrow(len(images))
        grid_image = make_grid(images, nrow, padding, normalize, value_range, scale_each, pad_value)
        pil_image = Image.fromarray(
            grid_image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
        if captions is not None:
            pil_image = draw_captions_over_image(
                pil_image, captions,
                grid_cell_size=(images[0].size(-2), images[0].size(-1)),
                grid_cell_padding=padding,
                caption_color=caption_color,
                caption_font=caption_font
            )
        pil_image.save(save_path)
        return

    if isinstance(save_path, (str, Path)):
        save_path = Path(save_path)
        save_path = [save_path.with_name(f"{save_path.stem}_{i}{save_path.suffix}") for i in range(len(images))]
    assert len(save_path) >= len(images)
    for i in range(len(images)):
        image = images[i:i + 1]
        caption = None if captions is None else captions[i:i + 1]
        image = make_grid(image, nrow, padding, normalize, value_range, scale_each, pad_value)
        pil_image = Image.fromarray(
            image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy())
        if caption is not None:
            pil_image = draw_captions_over_image(
                pil_image, captions,
                grid_cell_size=(image.size(-2), image.size(-1)),
                grid_cell_padding=padding,
                caption_color=caption_color,
                caption_font=caption_font
            )
        pil_image.save(save_path[i])


def images_files(image_folder, recursive=False):
    pattern = "**/*" if recursive else "*"
    root = Path(image_folder).resolve()
    if not root.exists():
        return []
    files = [file for file in root.glob(pattern) if is_image_file(file.name)]
    files = sorted(files, key=os.path.getmtime)
    return files


class LMDBCacheLoader:
    def __init__(self, lmdb_cache_path, loader=default_loader):
        self.loader = loader
        self.env = lmdb.open(
            lmdb_cache_path,
            map_size=1024 ** 4,
            readahead=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_cache_path)
        self.txn = self.env.begin(write=True)

    def __call__(self, file_path):
        assert isinstance(file_path, (str, Path))
        file_path = file_path if isinstance(file_path, str) else str(file_path)

        key = file_path.encode('utf-8')
        result_bytes = self.txn.get(key)
        if result_bytes is None:
            result = self.loader(file_path)
            # save loaded result to lmdb dataset
            buffer = BytesIO()
            pickle.dump(result, buffer)
            self.txn.put(key, buffer.getvalue())
            return result
        return pickle.load(BytesIO(result_bytes))
