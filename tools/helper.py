import os
from itertools import chain
from pathlib import Path

import cv2
import fire
import torch
from loguru import logger
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm

import raydl


def split(*image_path, save_path, n_height, n_width, resize=None):
    image = raydl.load_images(image_path)
    base_h, base_w = image.size()[-2] // n_height, image.size()[-1] // n_width
    print(f"original image size: {image.size()} base crop size: {(base_h, base_w)}")
    split_images = []
    for h in range(n_height):
        for w in range(n_width):
            split_images.append(image[:, :, h * base_h:(h + 1) * base_h, w * base_w:(w + 1) * base_w])
    split_images = torch.cat(split_images)
    raydl.save_images(split_images, save_path, resize=resize, separately=True)


def video(image_folder, save_path, fps=4, crop_width=None, crop_height=None, crop_pixel_base=1):
    files = [file for file in Path(image_folder).glob("*") if is_image_file(file.name)]
    files = sorted(files, key=os.path.getmtime)
    print([f.name for f in files])

    out_stream = None

    crop_width = crop_width * crop_pixel_base if crop_width is not None else None
    crop_height = crop_height * crop_pixel_base if crop_height is not None else None

    for f in tqdm(files):
        img = cv2.imread(f.as_posix())
        height, width, _ = img.shape
        height = crop_height or height
        width = crop_width or width
        size = (width, height)

        if out_stream is None:
            out_stream = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, size)
        out_stream.write(img[:height, :width, :])
    out_stream.release()


def diff(path1, path2, save_path=None, ada=False):
    path1 = Path(path1)
    path2 = Path(path2)

    if path1.is_file() and path2.is_file():
        save_path = Path(save_path) if save_path is not None else path1.parent
        images_iterator = [(path1, path2, save_path / f"{path1.stem}_diff_{path2.stem}{path1.suffix}")]
    elif path1.is_dir() and path2.is_dir():
        save_path = Path(save_path) if save_path is not None else path1.parent / f"{path1.name}_diff_{path2.name}"
        if not save_path.exists():
            print(f"mkdir {save_path}")
            save_path.mkdir()
        files1 = [f for f in Path(path1).glob("*") if is_image_file(f.name) and (path2 / f.name).exists()]
        files2 = [path2 / f.name for f in files1]
        images_iterator = zip(files1, files2, [save_path / f.name for f in files1])
    else:
        raise ValueError(f"{path1} and {path2} are not both files or both folders")

    for image_path1, image_path2, path in images_iterator:
        image1 = raydl.load_images(image_path1)
        image2 = raydl.load_images(image_path2)
        assert image1.size() == image2.size()
        range_max = (2 ** 2 * 3) ** 0.5 if not ada else None

        image_diff = torch.norm(image1 - image2, p=2, dim=1)
        print(f"{image_path1} and {image_path2} "
              f"max difference is {image_diff.max():.4f} ({100 * image_diff.max() / ((2 ** 2 * 3) ** 0.5):.2f}%)")
        diff_heatmap = raydl.create_heatmap(image_diff, scale_each=False, range_min=0, range_max=range_max)

        if path.exists():
            _path = path.parent / f"{path.stem}_diff{path.suffix}"
            print(f"{path} existed! rename save path to {_path}")
            path = _path
        raydl.save_images(diff_heatmap, path)


def concat(*image_folders, save_path, captions=None, resize=None, nrow=None, transpose=True, batch_size=None):
    image_folders = [Path(f) for f in image_folders]
    images_to_compare = list(set.intersection(*[
        set([p.name for p in chain(
            folder.glob("*.jpg"),
            folder.glob("*.png")
        )]) for folder in image_folders
    ]))
    if len(images_to_compare) == 0:
        print("can not found the same name jpg or png images in these image_folders")
        return
    images_to_compare = sorted(images_to_compare)
    logger.info(f"have total {len(images_to_compare)} images")
    logger.info(f"image_folders:\n" + '\n\t'.join([str(f) for f in image_folders]))

    batch_size = batch_size or len(images_to_compare)
    i = 0
    save_path = Path(save_path)

    for batch_id, batch in enumerate(raydl.total_chunk(len(images_to_compare), batch_size, drop_last=False)):
        images = torch.cat([
            raydl.load_images([folder / name for name in images_to_compare[i:i + batch]], resize=resize)
            for folder in image_folders
        ])
        i += batch

        if captions is not None:
            captions = list(map(str, captions))
            assert len(captions) == len(image_folders), f"{captions} v.s len(image_folders)={len(image_folders)}"
            if not transpose:
                captions = sum([[c, *[None] * (batch - 1)] for c in captions], [])
            else:
                captions.extend([None] * (len(captions) * (batch - 1)))
        default_nrow = batch
        if transpose:
            images = raydl.grid_transpose(images, batch)
            default_nrow = images.size(0) // default_nrow

        if batch_size == 1:
            sp = save_path.parent / images_to_compare[batch_id]
        elif batch_size != len(images_to_compare):
            sp = save_path.parent / f"{save_path.stem}_{batch_id}{save_path.suffix}"
        else:
            sp = save_path
        logger.info(sp)
        raydl.save_images(images, save_path=sp, captions=captions, resize=resize, nrow=nrow or default_nrow)


if __name__ == '__main__':
    fire.Fire()
