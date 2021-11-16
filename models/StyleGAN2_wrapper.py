from pathlib import Path

import torch
from loguru import logger
from torch.nn import Identity

from models.StyleGAN2 import Generator as StyleGAN2Generator


class ImprovedStyleGAN2Generator(StyleGAN2Generator):
    """
    wrap original StyleGAN Generator for manipulation.
    """

    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            channel_multiplier=2,
            blur_kernel=(1, 3, 3, 1),
            lr_mlp=0.01,
            default_truncation=0.7,
            num_fp16_res=0,
            conv_clamp=None,
    ):
        super().__init__(size, style_dim=style_dim, n_mlp=n_mlp, channel_multiplier=channel_multiplier,
                         blur_kernel=blur_kernel, lr_mlp=lr_mlp, num_fp16_res=num_fp16_res, conv_clamp=conv_clamp)

        self.default_truncation = default_truncation
        self.truncation_latent = None

        layers = [self.conv1, self.to_rgb1]
        for conv1, conv2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            layers += [conv1, conv2, to_rgb]

        self.layers = layers

    def manipulation_mode(self, flag=True):
        with torch.no_grad():
            if self.truncation_latent is None:
                self.truncation_latent = self.mean_latent(4096)
        for layer in self.layers:
            if flag:
                layer.conv.modulation, layer.conv.modulation_ = Identity(), layer.conv.modulation
            else:
                layer.conv.modulation = layer.conv.modulation_
                del layer.conv.modulation_
        logger.debug(f"enable manipulation mode: {flag}")

    def z_to_w(self, z, truncation=None, truncation_latent=None):
        w = self.style(z)
        truncation = truncation if truncation is not None else self.default_truncation
        truncation_latent = self.truncation_latent if truncation_latent is None else truncation_latent
        if truncation_latent is None:
            truncation_latent = self.mean_latent(4096)
            self.truncation_latent = truncation_latent
        if truncation < 1:
            w = truncation_latent + truncation * (w - truncation_latent)
        return w

    def w_to_styles(self, w):
        if w.dim() == 2:
            w = w.unsqueeze(dim=1).expand(-1, self.n_latent, self.style_dim)

        styles = [self.conv1.conv.modulation_(w[:, 0]), self.to_rgb1.conv.modulation_(w[:, 1])]

        i = 1
        for conv1, conv2, to_rgb in zip(self.convs[::2], self.convs[1::2], self.to_rgbs):
            styles.append(conv1.conv.modulation_(w[:, i]))
            styles.append(conv2.conv.modulation_(w[:, i + 1]))
            styles.append(to_rgb.conv.modulation_(w[:, i + 2]))
            i += 2

        return styles

    def styles_to_image(self, styles, noise=None, randomize_noise=False):
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        out = self.input(styles[0])
        out = self.conv1(out, styles[0], noise=noise[0])

        skip = self.to_rgb1(out, styles[1])

        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, styles[i], noise=noise1)
            out = conv2(out, styles[i + 1], noise=noise2)
            skip = to_rgb(out, styles[i + 2], skip)
            i += 3

        image = skip

        return image

    def styles_to_image_and_features(self, styles, layer_indexes, noise=None, randomize_noise=False):
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)]

        layer_indexes = [layer_indexes, ] if isinstance(layer_indexes, int) else layer_indexes
        features = []
        out = self.input(styles[0])
        if 0 in layer_indexes:
            features.append(out)
        out = self.conv1(out, styles[0], noise=noise[0])

        if 1 in layer_indexes:
            features.append(out)
        skip = self.to_rgb1(out, styles[1])

        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            if i in layer_indexes:
                features.append(out)
            out = conv1(out, styles[i], noise=noise1)
            if i + 1 in layer_indexes:
                features.append(out)
            out = conv2(out, styles[i + 1], noise=noise2)
            if i + 2 in layer_indexes:
                features.append(out)
            skip = to_rgb(out, styles[i + 2], skip)
            i += 3

        image = skip
        return image, features

    def w_to_image(self, w, noise=None, randomize_noise=False):
        styles = self.w_to_styles(w)
        return self.styles_to_image(styles, noise, randomize_noise)

    def z_to_image(self, z, truncation=None, truncation_latent=None, noise=None, randomize_noise=True):
        w = self.z_to_w(z, truncation, truncation_latent)
        return self.w_to_image(w, noise, randomize_noise)

    def forward(self, w=None, styles=None, z=None, modifications=None):  # noqa
        assert w is not None or styles is not None or z is not None
        styles = [s.clone() for s in styles] if styles is not None \
            else self.w_to_styles(w if w is not None else self.z_to_w(z))
        if modifications is not None:
            assert isinstance(modifications, (list, tuple))
            for mdfc, apply_param in modifications:
                styles = mdfc.apply(styles, **apply_param)
        image = self.styles_to_image(styles)
        return image

    @staticmethod
    def conv_layer_name(layer):
        if layer == 0:
            return "conv1"
        elif layer == 1:
            return "to_rgb1"
        elif layer % 3 == 1:
            return f"to_rgbs.{layer // 3 - 1}"
        else:
            return f"convs.{2 * ((layer - 2) // 3) + (layer + 1) % 3}"

    @staticmethod
    def parameter_tag(name: str):
        if name.startswith("style."):
            return "mapping"
        if "noise" in name:
            return "noise"
        if name.startswith("input."):
            return "input"
        if "modulation" in name:
            return "modulation"
        if "blur" in name:
            return "conv.blur"
        if "bias" in name:
            return "conv.bias"
        if "upsample" in name:
            return "to_rgb.upsample"
        if "weight" in name:
            return "conv.weight"
        raise ValueError("invalid param name: " + name)

    @staticmethod
    def infer_arguments(state_dict: dict):
        num_layers = len(list(filter(lambda x: x.startswith("noise"), state_dict.keys())))
        arguments = dict()
        arguments["size"] = 2 ** ((num_layers - 1) / 2 + 2)
        arguments["style_dim"] = state_dict["style.1.bias"].size()[0]
        arguments["n_mlp"] = len(list(filter(lambda x: x.startswith("style"), state_dict.keys()))) // 2
        arguments["channel_multiplier"] = state_dict["convs.8.conv.modulation.bias"].size()[0] // 256
        return arguments

    @staticmethod
    def load(model_path, lr_mlp=0.01, default_truncation=0.7, conv_clamp=256.0, num_fp16_res=4, device="cuda"):
        """
        load checkpoint. Will return a Generator instance
        """
        model_path = Path(model_path)
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        if (model_path.parent.parent / "config.yaml").exists():
            # find saved config yaml for this checkpoint
            # create generator use arguments in config.yaml
            from omegaconf import OmegaConf
            conf = OmegaConf.load(model_path.parent.parent / "config.yaml")
            arguments = conf.model.generator
            for key in ["_type", ]:
                arguments.pop(key)
            logger.info("use arguments from config.yaml")
            logger.info(arguments)
        else:
            arguments = ImprovedStyleGAN2Generator.infer_arguments(checkpoint["g_ema"])
            arguments["lr_mlp"] = lr_mlp
            arguments["num_fp16_res"] = num_fp16_res
            arguments["conv_clamp"] = conv_clamp

        arguments["default_truncation"] = default_truncation
        g_ema = ImprovedStyleGAN2Generator(**arguments).to(device)
        for k in list(checkpoint["g_ema"].keys()):
            if "noises" in k:
                checkpoint["g_ema"].pop(k)
        logger.warning(g_ema.load_state_dict(checkpoint["g_ema"], strict=False))

        if "latent_avg" in checkpoint:
            g_ema.truncation_latent = checkpoint["latent_avg"].unsqueeze(dim=0).to(device)
        return g_ema
