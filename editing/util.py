import subprocess
from pathlib import Path

import torch
from loguru import logger

import raydl
from editing.config import CORRECTION_PATH


def convert_images_to_video(image_pattern, save_path, fps=24, overwrite=True, crf=18):
    # install ffmpeg last version first. https://johnvansickle.com/ffmpeg/
    # will construct shell command like:
    # ffmpeg  -r 15 -f image2 -i tmp/sequence_%d.jpg -vcodec libx264 -crf 18 -pix_fmt yuv420p tmp/t4.mp4
    args = ["ffmpeg"]
    if overwrite:
        args.append("-y")
    args += [
        "-r",
        f"{fps}",
        "-f",
        "image2",
        "-i",
        f"{image_pattern}",
        "-vcodec",
        "libx264",
        "-crf",
        f"{crf}",
        "-pix_fmt",
        "yuv420p",
        save_path,
        "-hide_banner",
        "-loglevel",
        "warning",
    ]
    logger.info(" ".join(args))
    subprocess.run(args)


# StyleGAN2 specific utilize functions

def prev_layer(layer: int):
    """
    previous main layer of input layer in StyleGAN Generator
    :param layer: layer index
    :return: previous layer index
    """
    assert layer >= 0

    # to_rgb layer
    if layer % 3 == 1:
        return layer - 1
    if layer % 3 == 0:
        return layer - 1
    if layer % 3 == 2:
        return layer - 2


def next_layer(layer):
    """
    next main layer of input layer in StyleGAN Generator
    :param layer: layer index
    :return: next layer index
    """
    if layer == 0:
        return 1
    # to_rgb layer
    if layer % 3 == 1:
        return layer + 1
    if layer % 3 == 0:
        return layer + 2
    if layer % 3 == 2:
        return layer + 1


def channel_selector(layers, rules="all", is_indexes_rule=False, correction_path=CORRECTION_PATH) -> dict:
    """
    select channels according to rules.
    :param layers: selected layers.
    :param rules: python command that will be run with eval(). something like "ric[10]>0.1"
    or "(ric[10]>0.1)&(ric[8]>0.1)"
    :param is_indexes_rule: if set as True, directly convert rules as channel indices, like "12,1,511"
    :param correction_path: channel-region corrections.
    :return: dict(layer1=mask1, layer2=mask2)
    """
    correction = torch.load(correction_path)

    layers = raydl.tuple_of_indices(layers)
    if isinstance(rules, int) and is_indexes_rule:
        rules = f"{rules}"
    rules = raydl.tuple_of_type(rules, str)
    if len(rules) == 1:
        rules = rules * len(layers)
    assert len(layers) == len(rules)

    result = {}
    for layer, rule in zip(layers, rules):
        corr = correction[layer].abs()
        # relative correction by mask
        rcm = corr / corr.amax(dim=1, keepdim=True)  # noqa
        # relative correction by channel
        rcc = corr / corr.amax(dim=0, keepdim=True)  # noqa
        # relative importance by channel
        ric = corr / corr.sum(dim=0, keepdim=True)  # noqa
        if is_indexes_rule:
            mask = torch.zeros_like(corr[0])
            for c in raydl.parse_indices_str(rule):
                mask[c] = 1
        else:
            if rule != "all":
                # rule from user, very very dangerous!!
                mask = eval(rule)
            else:
                mask = torch.ones_like(corr[0])
        logger.info(f"layer {layer} {int(mask.sum())} dims: {torch.nonzero(mask.float()).flatten().tolist()[:50]}")
        assert mask.size() == torch.Size([corr.size(1)])
        result[layer] = mask

    return result


def load_latent(
        latent_path, ids=None, load_real=False, real_path=None,
        latent_type=None, real_images_resize=1024,
        device=torch.device("cuda")
) -> dict:
    """
    load latent code of StyleGAN, may load real images too.
    :param latent_path:
    :param ids: latent indices. something like "1,2,9"
    :param load_real: whether load real images while real images path can be found.
    :param real_path: real images path
    :param latent_type: if None, will infer from suffix, or treat as "w"
    :param real_images_resize: will resize loaded real images.
    :param device:
    :return: latent dict: dict(latent_type:latent)
    """
    latent_path = Path(latent_path)
    ids = raydl.tuple_of_indices(ids) if ids is not None else None
    real_images = None
    if latent_type is None:
        latent_type = latent_path.suffix[1:]

    if load_real:
        real_path = Path(real_path or latent_path.with_suffix(".realpath"))
        if not real_path.exists():
            logger.warning(f"can not find associated real images path")
        else:
            reals = torch.load(real_path)
            if ids is not None:
                reals = [reals[i] for i in ids if -len(reals) <= i < len(reals)]
            try:
                if (latent_path.parent / "real").exists():
                    reals = [(latent_path.parent / "real") / Path(r).name for r in reals]
                real_images = raydl.load_images(reals, resize=real_images_resize)
                logger.debug(f"load {len(real_images)} real images")
            except FileNotFoundError as e:
                logger.error(e)
                real_images = None

    latent = torch.load(latent_path)
    if isinstance(latent, (list, tuple)):
        latent_type = "styles"
        if ids is not None:
            ids = (i for i in ids if -len(latent[0]) <= i < len(latent[0]))
            latent = [
                s[ids, :] if s.dim() == 2 else s[ids, :, :]
                for s in latent
            ]
        latent = [s.to(device) for s in latent]
    else:
        if ids is not None:
            ids = [i for i in ids if -len(latent) <= i < len(latent)]
            latent = latent[ids, :, :] if latent.dim() == 3 else latent[ids, :]
        latent = latent.to(device)
    if latent_type not in ["w", "z", "styles"]:
        latent_type = "w" if latent.dim() == 3 else "z"
        logger.warning(f"do not set correctly latent type, set as infer type: {latent_type}")
    return {latent_type: latent} if real_images is None else {latent_type: latent, "image": real_images}
