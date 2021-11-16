#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
from itertools import chain
from pathlib import Path

import fire
import torch
from loguru import logger
from tqdm import tqdm

import raydl
from editing import util
from editing.config import CHECKPOINT_PATH
from editing.modification import Manipulator, ManipulatorBuilder, Mixer
from editing.util import load_latent, channel_selector
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator


def load_modifications(modifications, device=torch.device("cuda"), verbose=True):
    modifications = raydl.tuple_of_type(modifications, str)
    outputs = []
    for mdfc_str in modifications:
        m = mdfc_str.split("#")
        if len(m) == 1:
            if verbose:
                logger.warning(f"{mdfc_str} do not contains move_factor, use default value: 10")
            m = (m, 10)
        mdfc, mf, rf = m if len(m) == 3 else (*m, 0.0)
        outputs.append((Manipulator.load(mdfc, device), dict(move_factor=float(mf), replace_factor=float(rf))))
        if verbose:
            logger.debug(outputs[-1])
    return outputs


def batch_latents(batch_size=16, num_samples=32, device=torch.device("cuda"), seed=200, generate_way="random",
                  style_dim=512, latent_dict: dict = None, drop_last=False):
    assert num_samples is not None or generate_way == "latent"

    if generate_way == "random":
        torch.manual_seed(seed)
        for batch in raydl.total_chunk(num_samples, batch_size, drop_last):
            yield dict(z=torch.randn(batch, style_dim, device=device))

    elif generate_way == "legacy":
        import numpy as np
        np.random.seed(seed)
        for batch in raydl.total_chunk(num_samples, batch_size, drop_last):
            all_seeds = np.random.randint(0, 1000000, size=batch)
            z = np.stack([np.random.RandomState(seed).randn(512) for seed in all_seeds])  # [minibatch, component]
            z = torch.from_numpy(z).to(device, torch.float)
            yield dict(z=z)

    elif generate_way == "latent":
        assert latent_dict is not None
        real_images = latent_dict.pop("image", None)
        if real_images is not None:
            logger.info(f"found {len(real_images)} in latent_dict, will batch-ly yield it")
        latent_type, latent = list(latent_dict.items())[0]
        if latent_type == "styles":
            num_samples = num_samples or len(latent[0])
            assert num_samples <= len(latent[0]), f"expect at least {num_samples} latent codes," \
                                                  f" but got {len(latent[0])}"
            num_generated_samples = 0
            for batch in raydl.total_chunk(num_samples, batch_size, drop_last):
                styles = [s[num_generated_samples: num_generated_samples + batch] for s in latent]
                if real_images is not None:
                    image = real_images[num_generated_samples:num_generated_samples + batch]
                    yield dict(styles=styles, image=image)
                else:
                    yield dict(styles=styles)
                num_generated_samples += batch
        else:
            num_samples = num_samples or len(latent)
            assert num_samples <= len(latent), f"expect at least {num_samples} latent codes," \
                                               f" but got {len(latent)}"
            num_generated_samples = 0
            for batch in raydl.total_chunk(num_samples, batch_size, drop_last):
                code = latent[num_generated_samples: num_generated_samples + batch]
                if real_images is not None:
                    image = real_images[num_generated_samples:num_generated_samples + batch]
                    yield {"image": image, latent_type: code}
                else:
                    yield {latent_type: code}
                num_generated_samples += batch


class Worker:
    def __init__(self, ckp=CHECKPOINT_PATH, save_folder="./tmp", truncation=0.7, device="cuda"):
        """
        manipulation's related worker.
        :param ckp: checkpoint path of generator.
        :param save_folder: path to save generated results
        :param device: cuda
        """

        logger.debug(raydl.describe_dict(dict(
            checkpoint=ckp,
            save_folder=save_folder,
            truncation=truncation
        )))

        self.device = torch.device(device)
        self._G = None
        self.G = ImprovedStyleGAN2Generator.load(ckp, device=self.device, default_truncation=truncation)
        logger.debug(f"load generator from {ckp}")
        self.G.manipulation_mode()
        self.G.eval()
        self.save_folder = Path(save_folder)
        if not self.save_folder.exists():
            self.save_folder.mkdir()
            logger.info(f"create folder {self.save_folder}")

        torch.set_grad_enabled(False)

    def _styles(self, latent_dict):
        if "styles" in latent_dict:
            return latent_dict["styles"]
        if "w" in latent_dict:
            return self.G.w_to_styles(latent_dict["w"])
        if "z" in latent_dict:
            return self.G.w_to_styles(self.G.z_to_w(latent_dict["z"]))
        raise ValueError("have not valid latent in latent_dict")

    def _save_images(self, images, name, zero_images=None, captions=None, resize=None, nrow=None,
                     separately=False, transpose=False):
        dit = "jpg"
        generate_param = dict(resize=resize, nrow=nrow, separately=separately, captions=captions)
        if isinstance(images, list):
            num_col = len(images[0])
            if captions is not None and isinstance(captions, (list, tuple)) and len(captions) == len(images):
                if transpose:
                    generate_param["captions"] = captions * num_col
                else:
                    generate_param["captions"] = sum([[cap] * num_col for cap in captions], [])
            images = [raydl.resize_images(image, resize=resize) for image in images]
            images = torch.cat(images)
            if transpose:
                images = raydl.grid_transpose(images, original_nrow=num_col)
                num_col = len(images) // num_col
            generate_param["nrow"] = generate_param["nrow"] or num_col
        else:
            images = raydl.resize_images(images, resize)
        if zero_images is not None:
            if isinstance(zero_images, list):
                zero_images = torch.cat(zero_images)
            zero_images = raydl.resize_images(zero_images, resize=resize)

            assert zero_images.dim() == images.dim() == 4
            assert zero_images.size()[-3:] == images.size()[-3:], f"{zero_images.size()[-3:]} != {images.size()[-3:]}"
            assert images.size()[0] % zero_images.size()[0] == 0

            if images.size()[0] != zero_images.size()[0]:
                batch_size = images.size()[0]
                zero_images = zero_images.unsqueeze(dim=0).expand(batch_size // zero_images.size()[0], -1, -1, -1, -1)
                zero_images = zero_images.reshape(-1, *images.size()[-3:])
            diff_heatmap = raydl.create_heatmap(
                torch.norm(images - zero_images, p=2, dim=1),
                scale_each=False,
                range_min=0,
                range_max=(2 ** 2 * 3) ** 0.5,
            )
            raydl.save_images(diff_heatmap, self.save_folder / f"{name}_diff.{dit}", **generate_param)

        raydl.save_images(images, self.save_folder / f"{name}.{dit}", **generate_param)
        logger.info(f"save image at {self.save_folder / f'{name}.{dit}'}")

    def _save_sequence_images(self, images_iterator, name, save_type="grid", zero_images=None, sequence_captions=None,
                              resize=None, nrow=None, fps=None):
        if zero_images is not None:
            zero_images = zero_images.cpu()
        assert save_type in ["grid", "image", "video"]
        separately = save_type == "image"
        save_params = dict(resize=resize, nrow=nrow, separately=separately)
        if save_type == "grid":
            images_sequence = []
            for images in images_iterator:
                images_sequence.append(images.cpu())
            if sequence_captions is not None:
                captions = sum([[sequence_captions[i]] * len(images_sequence[i]) for i in range(len(images_sequence))],
                               [])
            else:
                captions = None
            self._save_images(
                images_sequence,
                name,
                captions=captions,
                zero_images=zero_images,
                **save_params
            )
        elif save_type == "image":
            for i, images in enumerate(images_iterator):
                self._save_images(
                    images.cpu(),
                    f"{name}_{i}",
                    zero_images=None,
                    captions=None,
                    **save_params
                )
        elif save_type == "video":
            save_params["nrow"] = None
            total = 0
            for i, images in enumerate(images_iterator):
                images = images.cpu()
                self._save_images(
                    images,
                    f"{name}_{i}",
                    zero_images=zero_images,
                    **save_params
                )
                total += 1
            util.convert_images_to_video(
                self.save_folder.as_posix() + f"/{name}_%d.jpg",
                self.save_folder.as_posix() + f"/{name}.mp4",
                fps=fps or total // 2,
            )
            if zero_images is not None:
                util.convert_images_to_video(
                    self.save_folder.as_posix() + f"/{name}_%d_diff.jpg",
                    self.save_folder.as_posix() + f"/{name}_diff.mp4",
                    fps=fps or total // 2,
                )

    def _apply_sequence_factors(self, mdfc, latent_dict, factors, another_factor, factor_type="move",
                                base_modification=None):
        if factor_type == "move":
            logger.info(f"replace_factor={another_factor} move_factor={factors}")
            modifications = [(mdfc, dict(move_factor=f, replace_factor=another_factor)) for f in factors]
        else:
            logger.info(f"move_factor={another_factor} replace_factor={factors}")
            modifications = [(mdfc, dict(move_factor=another_factor, replace_factor=f)) for f in factors]

        base_modification = base_modification if base_modification is not None else []

        for modification in modifications:
            yield self.G(**latent_dict, modifications=base_modification + [modification, ])

    def generate(self, *modifications, way="random", num_samples=None, batch_size=16, resize=512, latent_path=None,
                 separately=True, seed=200, load_real=False, captions=False):
        """
        generate separately images
        :param modifications: list of string to describe the modification
        :param way: `random` or `legacy`, if latent_path is specified, auto change the way to latent.
        :param num_samples: total generated samples, default is batch_size.
        :param batch_size: batch size, default is 32.
        :param resize: generated image size, default is 512.
        :param latent_path: if specified, use pre-computed latent, and set way as latent,
            set num_samples as None, which means all latent code
        :param separately: if set as False, save images as grid. Default is True.
        :param seed: random code for random way or legacy way. default is 200.
        :param load_real: if True, will also try to load the real images
        :param captions: if True, will generate idx caption for each image when separately=False.
        :return: None

        example:

        python3 manipulation.py generate --resize 1024 --latent_path  ./tmp_inversion/latents.w --separately 1
        python3 manipulation.py generate /path/to/mdfc#step_size --resize 1024 --latent_path  ./tmp_inversion/latents.w
        """
        modifications = load_modifications(modifications, device=self.device, verbose=True)
        latent_dict = None
        if latent_path is not None:
            way = "latent"
            latent_dict = load_latent(latent_path, ids=None, load_real=load_real, device=self.device)
        else:
            num_samples = num_samples or batch_size

        pbar = tqdm(
            batch_latents(
                batch_size, num_samples, self.device, seed=seed, generate_way=way,
                style_dim=512, latent_dict=latent_dict, drop_last=False
            ),
            total=(num_samples + batch_size - 1) // batch_size if num_samples is not None else None,
            ncols=80,
        )
        for batch_id, latent_dict in enumerate(pbar):
            real_images = latent_dict.pop("image", None)
            images = self.G(**latent_dict, modifications=modifications)
            if real_images is not None and not separately:
                images = torch.cat([real_images, images])
            raydl.save_images(
                images,
                [self.save_folder / f"{batch_id * batch_size + i}.jpg" for i in range(len(images))] if separately else
                self.save_folder / f"b{batch_id}.jpg",
                resize=resize,
                separately=separately,
                nrow=None if real_images is None else len(real_images),
                captions=[f"{batch_id * batch_size + i}" for i in range(len(images))] if (
                        captions and not separately) else None
            )

    def test(self, mdfc=None, max_factor=None, build_mdfc_way=None, base_modification=None,
             another_factor=0, start_factor=None, num_factors=5, paired_factor=True, factor_type="move",
             way="random", num_samples=None, latent_path=None, real_path=None, load_real=True, ids=None,
             batch_size=16, resize=256, seed=1010, save_type="grid", save_name=None, nrow=None, captions=True,
             enable_seed_increment=False, **build_mdfc_kwargs):
        """
        test different modifications or build modification file on the fly.

        1. test certain mdfc file

        ```
        python3 manipulation.py test /path/to/mdfc --max_factor 8 --ckp /path/to/ckp --batch_size 8
        ```

        this command will generate 8 row images with move_factor=[-8.0, -4.0, 0.0, 4.0, 8.0].

        add more option like

        `--paired_factor False` to make sure move_factor start with 0
        `--start_factor 4` to make sure move_factor start with 4
        `--latent_path /path/to/latent_code` means apply modifications on prepared latent codes.
        `--resize 256` to resize each generated images as 256*256
        `--num_samples 32` to generate 32 samples

        2. create new mdfc file

        ```
        python3 manipulation.py test --max_factor -2 --batch_size 8 --paired_factor 0 \
        --build_mdfc_way remove --mdfc_path /path/to/mdfc --layers 5,6 --rules "ric[6]>0.2"
        ```

        Use --build_mdfc_way to specific class method of ManipulatorBuilder and use `--key value` to specific kwargs
        for this class method. The above command will disable the change to channels that meet
        the requirements (channels that "ric[6]>0.2" and in 5-th or 6-th layers).

        ```
        python3 manipulation.py test --max_factor -2 --batch_size 8 --paired_factor 0 \
        --build_mdfc_way alter --mdfc_path /path/to/mdfc --layers 5,6 --rules "ric[6]>0.2"
        ```
        this command will only enable the change to channels that meet the requirements.

        You can further use the command to save the new mdfc file create by you.
        ```
        PYTHONPATH=. python3 editing/modification.py alter /path/to/mdfc --layers 5,6 --rules "ric[6]>0.2" \
            --mdfc_name /path/to/mdfc/save/path
        ```

        As you see, just copy the options for test to save the mdfc.

        """

        assert max_factor is not None, "Please specify max_factor!"
        assert mdfc is not None or build_mdfc_way is not None, \
            "Please specify mdfc path or the way to build mdfc online!"
        assert build_mdfc_way is None or isinstance(build_mdfc_way, str), \
            f"build_mdfc_way must be str to represent the method name " \
            f"of ManipulatorBuilder, but got {type(build_mdfc_way)}"

        if mdfc is not None:
            if isinstance(mdfc, (Path, str)):
                save_name = save_name or Path(mdfc).stem
            mdfcs = [Manipulator.load(mdfc, self.device), ]
            save_name = save_name or "modification"
            save_names = [save_name, ]
            mdfc_names_iterator = zip(mdfcs, save_names)
            if len(build_mdfc_kwargs):
                logger.warning(raydl.describe_dict(build_mdfc_kwargs, head="useless args:"))
        else:
            mdfc_builder = ManipulatorBuilder(save=False, device=self.device, generator=self.G)
            assert hasattr(mdfc_builder, build_mdfc_way)
            mdfc_names_iterator = getattr(mdfc_builder, build_mdfc_way)(**build_mdfc_kwargs)

        latent_dict = None
        if latent_path is not None:
            way = "latent"
            latent_dict = load_latent(latent_path, ids=ids, load_real=load_real, real_path=real_path,
                                      device=self.device, real_images_resize=resize)
        else:
            num_samples = num_samples or batch_size

        factors = raydl.factors_sequence(
            max_factor, num_factors=num_factors, start_factor=start_factor, paired_factor=paired_factor)

        if base_modification is not None:
            base_modification = load_modifications(base_modification, self.device)

        for mdfc_id, (mdfc, name) in enumerate(mdfc_names_iterator):
            logger.info("*" * 80)
            logger.info(f"modification: {mdfc}")
            for batch_id, latent_dict in enumerate(batch_latents(
                    batch_size,
                    num_samples,
                    self.device,
                    seed=seed + mdfc_id if enable_seed_increment else seed,
                    generate_way=way,
                    style_dim=512,
                    latent_dict=latent_dict,
                    drop_last=False)
            ):
                logger.info(f"batch: {batch_id}")
                real_images = latent_dict.pop("image", None)
                zero_images = self.G(**latent_dict) if real_images is None else real_images
                images_iterator = chain(
                    [real_images.cpu()] if real_images is not None else [],
                    self._apply_sequence_factors(
                        mdfc, latent_dict, factors, another_factor,
                        factor_type, base_modification=base_modification
                    ),
                )
                default_captions = ([] if real_images is None else ["real"]) + list(map(lambda x: f"{x:.2f}", factors))
                captions = captions if isinstance(captions, (list, tuple)) else (default_captions if captions else None)
                name = save_name or Path(name).name
                self._save_sequence_images(
                    images_iterator,
                    f"{name}_mdfc{mdfc_id}_batch{batch_id}",
                    save_type=save_type,
                    zero_images=zero_images,
                    sequence_captions=captions,
                    resize=resize,
                    nrow=nrow or len(zero_images),
                )

    def checkpoints(
            self, *checkpoints, way="random", num_samples=None, latent_path=None, real_path=None,
            load_real=True, ids=None, batch_size=32, resize=512, seed=1010, save_name=None, nrow=None, captions=True
    ):
        """
        Generate samples using the same latent code but different checkpoint, useful when debug finetuned models.
        """
        latent_dict_original = None
        if latent_path is not None:
            way = "latent"
            latent_dict_original = load_latent(
                latent_path, ids=ids, load_real=load_real, real_path=real_path,
                device=self.device, real_images_resize=resize
            )
        else:
            num_samples = num_samples or batch_size

        image_dict = defaultdict(list)
        real_images_list = []
        zero_images_list = []

        for batch_id, latent_dict in enumerate(batch_latents(
                batch_size,
                num_samples,
                self.device,
                seed=seed,
                generate_way=way,
                style_dim=512,
                latent_dict=latent_dict_original,
                drop_last=False
        )):
            reals = latent_dict.pop("image", None)
            images = self.G(**latent_dict)
            if reals is not None:
                real_images_list.append(reals.cpu())
            zero_images_list.append(images.cpu())

        default_captions = ["real", "recon"] if len(real_images_list) > 0 else ["original"]

        for ckp in checkpoints:
            name = Path(ckp).stem[-8:]
            G = ImprovedStyleGAN2Generator.load(ckp, device=self.device)
            G.manipulation_mode()
            for batch_id, latent_dict in enumerate(batch_latents(
                    batch_size,
                    num_samples,
                    self.device,
                    seed=seed,
                    generate_way=way,
                    style_dim=512,
                    latent_dict=latent_dict_original,
                    drop_last=False
            )):
                latent_dict.pop("image", None)
                images = G(**latent_dict)
                image_dict[batch_id].append(images.cpu())
            default_captions.append(name)

        captions = captions if isinstance(captions, (list, tuple)) else (default_captions if captions else False)

        for i in range(len(zero_images_list)):
            real = real_images_list[i] if len(real_images_list) > 0 else None
            zero = zero_images_list[i]
            base_list = [real, zero] if real is not None else [zero]
            self._save_images(
                [*base_list, *image_dict[i]],
                f"{save_name}_{i}",
                captions=captions,
                resize=resize,
                nrow=nrow or len(zero),
                separately=False,
                transpose=False
            )

    def mix(self, *modifications, latent1, ids1, ids2, latent2=None, batch_size=16, resize=512, load_real=True,
            mix_weight=0.5, mix_file=None, save_name="mix", transpose=False, **select_kwargs):
        """
        mix latents use mix file or cli kwargs, generate mixed images.
        """
        modifications = load_modifications(modifications, device=self.device, verbose=True)
        latent_dict1 = load_latent(latent1, ids1, load_real=load_real, device=self.device)
        latent2 = latent2 or latent1
        latent_dict2 = load_latent(latent2, ids2, load_real=load_real, device=self.device)
        real_images1 = latent_dict1.get("image", None)
        real_images2 = latent_dict2.get("image", None)
        styles1 = self._styles(latent_dict1)
        styles2 = self._styles(latent_dict2)
        assert len(styles1[0]) == len(styles1[0]), f"except same amount of latent for 1, 2"

        mix_weight = raydl.tuple_of_type(mix_weight, (int, float))

        for mw in mix_weight:
            if mix_file is not None:
                if "#" not in mix_file:
                    mixer = f"{mix_file}#{mw}"
                else:
                    mixer = mix_file
                mixer = Mixer.load(mixer)
            elif len(select_kwargs) > 0:
                mixer = Mixer(mix_weight=mw, select=channel_selector(**select_kwargs))
            else:
                mixer = Mixer(mix_weight=mw)

            for batch_id, (ld1, ld2) in enumerate(zip(
                    batch_latents(batch_size, device=self.device, generate_way="latent",
                                  latent_dict=dict(styles=styles1, image=real_images1), num_samples=len(styles1[0])),
                    batch_latents(batch_size, device=self.device, generate_way="latent",
                                  latent_dict=dict(styles=styles2, image=real_images2), num_samples=len(styles1[0])),
            )):
                results = [
                    (image, caption)
                    for image, caption in zip([ld1.get("image", None), ld2.get("image", None)], ["realA", "realB"])
                    if image is not None
                ]
                styles = mixer.apply(ld1["styles"], ld2["styles"])
                mix_images = self.G(styles=styles)
                final_images = self.G(styles=styles, modifications=modifications)
                results.append((mix_images, f"mix_{mw:.2f}"))
                results.append((final_images, f"result_{mw:.2f}"))

                self._save_images(
                    [image for image, _ in results],
                    f"{save_name}_{mw:.2f}_b{batch_id}",
                    captions=[cap for _, cap in results],
                    resize=resize,
                    transpose=transpose
                )

    def random_walk(self, num_samples=4, seed=1010, num_frames=48, save_name="random_walk", resize=512, fps=None):
        """
        generate a smooth latent code walk in the W space video.
        """
        torch.manual_seed(seed)
        z = torch.randn(num_samples, 512, device=self.device)

        w = self.G.z_to_w(torch.cat([z, z[0:1]]))

        def walker():
            for lid in range(num_samples):
                w_start, w_end = w[lid:lid + 1], w[lid + 1:lid + 2]
                move_factors = raydl.factors_sequence(end_factor=1, start_factor=0, num_factors=num_frames)
                for f in move_factors:
                    yield self.G(w=w_start + (w_end - w_start) * f)

        self._save_sequence_images(walker(), save_name, save_type="video", resize=resize, fps=fps or num_frames // 4)

    def style_mixing(self, layers, ncols=4, nrow=8, seed=1000, resize=256):
        layers = raydl.tuple_of_indices(layers)
        torch.manual_seed(seed)
        styles1 = self.G.w_to_styles(self.G.z_to_w(torch.randn(nrow, 512, device=self.device)))
        styles2 = self.G.w_to_styles(self.G.z_to_w(torch.randn(ncols, 512, device=self.device)))
        head = self.G.styles_to_image(styles1)
        head = torch.cat([torch.ones(1, *head.shape[1:], device=self.device), head])

        for layer in tqdm(layers, ncols=80):
            images = [head]
            for i in range(ncols):
                style_h = [s[i:i + 1] for s in styles2]
                style = [(s.clone() if l < layer else style_h[l].expand(nrow, -1)) for l, s in enumerate(styles1)]
                style = [torch.cat([s1, s2]) for s1, s2 in zip(style_h, style)]
                images.append(self.G.styles_to_image(style))
            captions = [f"layer>={layer}", *[None] * ((nrow + 1) * (ncols + 1) - 1)]
            raydl.save_images(images, self.save_folder / f"style_mixing_{layer}.jpg", nrow=nrow + 1, resize=resize,
                              captions=captions)


if __name__ == '__main__':
    try:
        torch.set_grad_enabled(False)
        fire.Fire(Worker)
    except Exception as e:  # noqa
        # try to catch all Exception and log it.
        logger.exception(e)
