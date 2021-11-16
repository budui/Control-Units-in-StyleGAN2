# modified from https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/training/networks.py

# NVIDIA LICENSE:
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# author: Ray Wang <wangbudui@foxmail.com>

import math
import random

import numpy as np
import torch
import torch.nn as nn

import raydl
from models import MODEL
from models.ada_ops import (
    conv2d_resample,
    fma,
    bias_act,
    upfirdn2d
)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def modulated_conv2d(
        x, weight: nn.Parameter, style, noise=None, up=1, down=1, padding=0,
        resample_filter=None, demodulate=True,
        flip_weight=True, fused_modconv=True,
        weight_gain=1.0,
) -> torch.Tensor:
    """
    conv2d with modulated conv weights.
    :param x: input feature. BxCxHxW
    :param weight: conv weight. OxIxKhxKw
    :param style: modulation style vector. BxI or BxOxI (filter-wise styles)
    :param noise: optional noise tensor to add to the output activations.
    :param up: integer upsampling factor.
    :param down: integer downsampling factor.
    :param padding: padding with respect to the upsampled image.
    :param resample_filter: low-pass filter to apply when resampling activations.
    must be prepared beforehand by calling upfirdn2d.setup_filter().
    :param demodulate: apply weight demodulation?
    :param flip_weight: False = convolution, True = correlation (matches torch.nn.functional.conv2d)
    :param fused_modconv: perform modulation, convolution, and demodulation as a single fused operation?
    :param weight_gain: will mul weight after demodulation.
    :return: output feature maps. BxCx(up*H)x(up*W) when used in Generator.
    """
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.size()

    # pre-normalizing the style vector s and each row of the weight tensor w
    # before applying weight modulation and demodulation
    # from StyleGAN2-ADA. appendix.D.1: Mixed-precision training.
    if x.dtype == torch.float16 and demodulate:
        # During demodulation, each filter have in_channels * kh * kw elements,
        # which are multiplied by style and then squared to sum.
        # so, pre-normalize like this to make sure the sum is smaller than 1.
        # filter / filter.max() / num_of_elements(filter)
        weight = weight * (1 / math.sqrt(in_channels * kh * kw) /
                           weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))
        # style / style.max()
        style = style / style.norm(float('inf'), dim=1, keepdim=True)

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # 1xOxIxKhxKw
        w = w * style.reshape(batch_size, -1, in_channels, 1, 1)  # Bx(1|O)xIx1x1
    if demodulate:
        dcoefs = (w.square().sum(dim=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    # use modulation style to modulate feature maps, then normalize it with dcoefs
    if not fused_modconv:
        x = x * style.to(x.dtype).reshape(batch_size, -1, 1, 1)
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight.mul(weight_gain).to(x.dtype), f=resample_filter, up=up, down=down,
            padding=padding, flip_weight=flip_weight
        )
        if demodulate and noise is not None:
            # fused multiply & add
            x = fma.fma(
                x,
                dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1),
                noise.to(x.dtype)
            )
        elif demodulate:
            x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
        elif noise is not None:
            x = x.add_(noise.to(x.dtype))
        return x

    # first construct modulated conv weights, than conv.
    x = x.reshape(1, -1, *x.shape[2:])  # 1x(BxC)xHxW
    w = w.reshape(-1, in_channels, kh, kw)  # (BxC)xIxKhxKw
    x = conv2d_resample.conv2d_resample(
        x=x, w=w.mul(weight_gain).to(x.dtype), f=resample_filter, up=up, down=down, padding=padding,
        groups=batch_size,  # use group-wised conv to calculate batch results.
        flip_weight=flip_weight
    )
    x = x.reshape(batch_size, -1, *x.shape[2:])
    if noise is not None:
        x = x.add_(noise)
    return x


class FusedBiasActivation(nn.Module):
    def __init__(self, activation, clamp=None):
        super().__init__()
        assert activation in bias_act.activation_funcs, \
            f"invalid activation function, " \
            f"only support {list(bias_act.activation_funcs.keys())}, " \
            f"but got {activation}"

        self.activation = activation
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.clamp = clamp

    def __repr__(self):
        return raydl.class_repr(self, ["activation", "clamp"])

    def forward(self, x, bias=None, gain=1):
        return bias_act.bias_act(
            x,
            bias.to(x.dtype) if bias is not None else None,
            act=self.activation,
            gain=self.act_gain * gain,
            clamp=self.clamp * gain if self.clamp is not None else None,
        )


class EqualFullyConnectedLayer(nn.Module):
    """
    Idea from PGGAN Sec4.1 EQUALIZED LEARNING RATE
    trick 1: scale activation with kaiming constant every forward rather than only in weight initialization
    trick 2: learning rate multiplier
    """

    def __init__(self, in_features, out_features, bias=True, activation='linear',
                 lr_multiplier=1.0, bias_init=0.0):
        """
        Equal Fully Connected Layer
        :param in_features: number of input features.
        :param out_features: number of output features.
        :param bias: apply additive bias before the activation function?
        :param activation: activation function: 'relu', 'lrelu', etc.
        :param lr_multiplier: learning rate multiplier.
        :param bias_init: initial value for the additive bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation

        # use a trivial normal initialization
        # and the weight will be explicitly scaled at runtime
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]).div(lr_multiplier))
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None

        self.lr_multiplier = lr_multiplier
        self.weight_gain = lr_multiplier / math.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def __repr__(self):
        return raydl.class_repr(self, ["in_features", "out_features", "activation", "lr_multiplier"])

    def forward(self, x: torch.Tensor):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b.mul(self.bias_gain)

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, w_dim, c_dim, num_latent=None, num_layers=8,
                 embed_features=None, layer_features=None, activation='lrelu',
                 lr_multiplier=0.01):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_latent = num_latent
        self.num_layers = num_layers

        self.embed_features = 0 if c_dim == 0 else (embed_features or w_dim)
        self.layer_features = layer_features or w_dim

        self.embed = EqualFullyConnectedLayer(c_dim, embed_features) if c_dim > 0 else None

        features_list = [z_dim + self.embed_features] + [self.layer_features] * (num_layers - 1) + [w_dim]

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = EqualFullyConnectedLayer(
                in_features, out_features,
                activation=activation, lr_multiplier=lr_multiplier
            )
            setattr(self, f'fc{idx}', layer)

    def forward(self, z_list, c=None, truncation=1, truncation_latent=None, truncation_cutoff=None, inject_index=None):
        z_list = z_list if isinstance(z_list, (list, tuple)) else [z_list, ]
        w_list = []
        for z in z_list:
            x = None
            if self.z_dim > 0:
                x = normalize_2nd_moment(z.to(torch.float32))
            if c is not None and self.c_dim > 0:
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

            for idx in range(self.num_layers):
                x = getattr(self, f"fc{idx}")(x)
            w_list.append(x)

        if self.num_latent is not None:
            if len(w_list) == 1:
                x = w_list[0].unsqueeze(1).repeat([1, self.num_latent, 1])
            else:
                if inject_index is None:
                    inject_index = random.randint(1, self.num_latent - 1)
                x = torch.cat([
                    w_list[0].unsqueeze(1).repeat([1, inject_index, 1]),
                    w_list[1].unsqueeze(1).repeat([1, self.num_latent - inject_index, 1])
                ], dim=1)
        else:
            assert len(w_list) == 1
            x = w_list[0]

        if 0 <= truncation < 1:
            if self.num_latent is None or truncation_cutoff is None:
                x = truncation_latent.lerp(x, truncation)
            else:
                x[:, :truncation_cutoff] = truncation_latent.lerp(x[:, :truncation_cutoff], truncation)
        return x


class SynthesisConv(nn.Module):
    def __init__(
            self, in_channels, out_channels, w_dim, resolution, kernel_size,
            up=1, use_noise=True, activation="lrelu", resample_filter=(1, 3, 3, 1),
            conv_clamp=None, channels_last=False):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2

        self.affine = EqualFullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        if use_noise:
            self.register_buffer('noise_const', torch.randn([resolution, resolution]))
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))

        self.activate = FusedBiasActivation(activation=activation, clamp=conv_clamp)

    def forward(self, x, w, noise_mode="random", fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        style = self.affine(w)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = torch.randn(
                [x.shape[0], 1, self.resolution, self.resolution], device=x.device) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1)  # slightly faster
        x = modulated_conv2d(
            x, self.weight, style, noise=noise, up=self.up, down=1, padding=self.padding,
            resample_filter=self.resample_filter, demodulate=True,
            flip_weight=flip_weight, fused_modconv=fused_modconv
        )
        return self.activate(x, self.bias, gain=gain)


class ToRGBLayer(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim,
                 kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()

        self.affine = EqualFullyConnectedLayer(w_dim, in_channels, bias_init=1)
        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        self.activate = FusedBiasActivation(activation="linear", clamp=conv_clamp)

    def forward(self, x, w, fused_modconv=True):
        style = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, style=style, demodulate=False, fused_modconv=fused_modconv)
        return self.activate(x, self.bias)


class ResolutionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, w_dim, resolution, img_channels,
                 is_last, resample_filter=(1, 3, 3, 1), conv_clamp=None, use_fp16=False,
                 fp16_channels_last=False):
        super().__init__()

        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)

        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))

        layer_kwargs = dict(use_noise=True, activation="lrelu", kernel_size=3)

        if in_channels == 0:
            self.const = torch.nn.Parameter(torch.randn([out_channels, resolution, resolution]))
        else:
            self.conv0 = SynthesisConv(
                in_channels, out_channels, w_dim=w_dim, resolution=resolution,
                up=2, resample_filter=resample_filter, conv_clamp=conv_clamp,
                channels_last=self.channels_last, **layer_kwargs
            )

        self.conv1 = SynthesisConv(
            out_channels, out_channels, w_dim=w_dim, resolution=resolution, up=1,
            resample_filter=resample_filter, conv_clamp=conv_clamp,
            channels_last=self.channels_last, **layer_kwargs
        )

        self.torgb = ToRGBLayer(
            out_channels, img_channels, w_dim=w_dim,
            conv_clamp=conv_clamp, channels_last=self.channels_last
        )

    def forward(self, x, image, latents, force_fp32=False, fused_modconv=None, noise_mode="random"):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == torch.float32 or int(x.shape[0]) == 1)

        assert isinstance(latents, (list, tuple)) and len(latents) == (2 if self.in_channels == 0 else 3)

        if self.in_channels == 0 and x is None:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([latents[0].size(0), 1, 1, 1])
        else:
            x = x.to(dtype=dtype, memory_format=memory_format)  # BxCx(self.resolution//2)x(self.resolution//2)

        layer_kwargs = dict(noise_mode=noise_mode, gain=1, fused_modconv=fused_modconv)
        if self.in_channels == 0:
            x = self.conv1(x, latents[0], **layer_kwargs)
        else:
            x = self.conv0(x, latents[0], **layer_kwargs)
            x = self.conv1(x, latents[1], **layer_kwargs)

        if image is not None:
            image = upfirdn2d.upsample2d(image, self.resample_filter)

        y = self.torgb(x, latents[-1], fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        image = image.add_(y) if image is not None else y

        assert x.dtype == dtype
        assert image is None or image.dtype == torch.float32
        return x, image


class SynthesisNetwork(nn.Module):
    def __init__(self, w_dim, img_resolution, img_channels, channel_multiplier=2,
                 num_fp16_res=0, resample_filter=(1, 3, 3, 1), conv_clamp=None):
        super().__init__()
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]

        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        block_kwargs = dict(
            resample_filter=resample_filter,
            conv_clamp=conv_clamp,
            fp16_channels_last=False
        )
        for res in self.block_resolutions:
            in_channels = self.channels[res // 2] if res > 4 else 0
            out_channels = self.channels[res]
            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = ResolutionBlock(
                in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16,
                **block_kwargs)
            setattr(self, f'b{res}', block)

        self.num_latents = 2 * len(self.block_resolutions)

    def forward(self, latents, force_fp32=False, fused_modconv=None, noise_mode="random", customized_const=None):
        num_conv_list = [1] + [2] * (len(self.block_resolutions) - 1)
        if torch.is_tensor(latents):
            if latents.dim() == 2:
                latents = latents.unsqueeze(dim=1).repeat(1, self.num_latents, 1)
            latents = latents.to(torch.float32)  # B x num_of_latents x D
            latents = latents.unbind(dim=1)
            i = 0
            _resolution_latents = []
            for idx, _ in enumerate(self.block_resolutions):
                _resolution_latents += latents[i:i + num_conv_list[idx] + 1]
                i += num_conv_list[idx]
        else:
            _resolution_latents = latents

        block_kwargs = dict(force_fp32=force_fp32, fused_modconv=fused_modconv, noise_mode=noise_mode)
        x = customized_const
        image = None
        i = 0
        for res, num_conv in zip(self.block_resolutions, num_conv_list):
            block = getattr(self, f"b{res}")
            x, image = block(x, image, _resolution_latents[i:i + num_conv + 1], **block_kwargs)
            i = num_conv + 1 + i
        return image


@MODEL.register_module("StyleGAN2-Generator-mine")
class Generator(nn.Module):
    def __init__(
            self,
            size,
            style_dim=512,
            c_dim=0,
            n_mlp=8,
            img_channels=3,
            channel_multiplier=2,
            blur_kernel=(1, 3, 3, 1),
            lr_mlp=0.01,
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            num_fp16_res=0,  # Use FP16 for the N highest resolutions.
            mapping_kwargs=None,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.synthesis = SynthesisNetwork(
            style_dim, size, img_channels, channel_multiplier=channel_multiplier,
            num_fp16_res=num_fp16_res, resample_filter=blur_kernel, conv_clamp=conv_clamp)
        self.num_latents = self.synthesis.num_latents

        mapping_kwargs = mapping_kwargs or dict(
            embed_features=None, layer_features=None, activation='lrelu'
        )
        self.mapping = MappingNetwork(
            style_dim, style_dim, c_dim, num_latent=self.num_latents, num_layers=n_mlp,
            lr_multiplier=lr_mlp, **mapping_kwargs
        )

    @torch.no_grad()
    def mean_latent(self, n_latent=4096):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.synthesis.b4.conv1.weight.device
        )
        latent = self.mapping(latent_in).mean(0, keepdim=True)[:, 0, :]

        return latent

    def forward(self, z, c=None, return_latents=False, inject_index=None,
                truncation=1, truncation_latent=None, truncation_cutoff=None,
                force_fp32=False, fused_modconv=None, noise_mode="random"):
        latents = self.mapping(z, c, truncation=truncation, truncation_latent=truncation_latent,
                               truncation_cutoff=truncation_cutoff, inject_index=inject_index)
        image = self.synthesis(latents, force_fp32=force_fp32, fused_modconv=fused_modconv, noise_mode=noise_mode)
        if return_latents:
            return image, latents
        return image, None


class Conv2DLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, activation="linear",
                 up=1, down=1, resample_filter=(1, 3, 3, 1), conv_clamp=None, channels_last=False):
        super(Conv2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        if resample_filter is not None:
            self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        else:
            self.resample_filter = None
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.activate = FusedBiasActivation(activation, clamp=conv_clamp)

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        self.weight = nn.Parameter(
            torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        )
        self.bias = nn.Parameter(torch.zeros([out_channels])) if bias else None

    def forward(self, x, gain=1):
        weight = self.weight.mul(self.weight_gain).to(x.dtype)
        flip_weight = (self.up == 1)
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight, f=self.resample_filter, up=self.up, down=self.down,
            padding=self.padding, flip_weight=flip_weight
        )
        return self.activate(x, self.bias, gain=gain)


class DiscriminatorBlock(nn.Module):
    def __init__(
            self, in_channels, tmp_channels, out_channels, resolution, img_channels, activation="lrelu",
            resample_filter=(1, 3, 3, 1), conv_clamp=None, use_fp16=False, fp16_channels_last=False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.img_channels = img_channels
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)

        if in_channels == 0:
            self.fromrgb = Conv2DLayer(
                img_channels, tmp_channels, kernel_size=1, activation=activation,
                conv_clamp=conv_clamp, channels_last=self.channels_last
            )

        self.conv0 = Conv2DLayer(
            tmp_channels, tmp_channels, kernel_size=3, activation=activation,
            conv_clamp=conv_clamp, channels_last=self.channels_last
        )
        self.conv1 = Conv2DLayer(
            tmp_channels, out_channels, kernel_size=3, activation=activation, down=2,
            resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last
        )
        self.skip = Conv2DLayer(
            tmp_channels, out_channels, kernel_size=1, bias=False, down=2,
            resample_filter=resample_filter, channels_last=self.channels_last
        )

        self.he_const = np.sqrt(0.5)

    def forward(self, x, force_fp32=False):
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.in_channels == 0:
            assert x.size(1) == self.fromrgb.in_channels
            x = self.fromrgb(x)
        y = self.skip(x, gain=self.he_const)
        x = self.conv1(self.conv0(x), gain=self.he_const)
        x = y.add_(x)

        assert x.dtype == dtype
        return x


class MinibatchStdLayer(torch.nn.Module):
    def __init__(self, group_size, num_channels=1):
        super().__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = torch.min(torch.as_tensor(self.group_size), torch.as_tensor(N)) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = x.reshape(G, -1, F, c, H, W)
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class DiscriminatorEpilogue(torch.nn.Module):
    def __init__(self, in_channels, cmap_dim, resolution, img_channels, mbstd_group_size=4,
                 mbstd_num_channels=1, activation='lrelu', conv_clamp=None):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels

        self.mbstd = MinibatchStdLayer(
            group_size=mbstd_group_size, num_channels=mbstd_num_channels
        ) if mbstd_num_channels > 0 else None
        self.conv = Conv2DLayer(
            in_channels + mbstd_num_channels, in_channels, kernel_size=3,
            activation=activation, conv_clamp=conv_clamp
        )
        self.fc = EqualFullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        self.out = EqualFullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)

    def forward(self, x, cmap):
        dtype = torch.float32
        memory_format = torch.contiguous_format

        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        x = self.fc(x.flatten(1))
        x = self.out(x)

        # Conditioning.
        if self.cmap_dim > 0:
            x = (x * cmap).sum(dim=1, keepdim=True) * (1 / (self.cmap_dim ** 0.5))

        assert x.dtype == dtype
        return x


@MODEL.register_module("StyleGAN2-Discriminator-mine")
class Discriminator(nn.Module):
    def __init__(
            self, size, channel_multiplier=2, num_image_channel=3, blur_kernel=(1, 3, 3, 1),
            activation='lrelu', stddev_group=4, num_fp16_res=0, conv_clamp=None,
            c_dim=0, cmap_dim=None, fp16_channels_last=False,
    ):
        super(Discriminator, self).__init__()
        self.c_dim = c_dim
        self.img_resolution = size
        self.img_resolution_log2 = int(np.log2(size))
        self.img_channels = num_image_channel
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        cmap_dim = 0 if c_dim == 0 else (cmap_dim or channels[4])

        for res in self.block_resolutions:
            in_channels = channels[res] if res < self.img_resolution else 0
            tmp_channels = channels[res]
            out_channels = channels[res // 2]
            use_fp16 = (res >= fp16_resolution)

            block = DiscriminatorBlock(
                in_channels, tmp_channels, out_channels, resolution=res, use_fp16=use_fp16,
                img_channels=self.img_channels, activation=activation,
                resample_filter=blur_kernel, conv_clamp=conv_clamp,
                fp16_channels_last=fp16_channels_last,
            )
            setattr(self, f'b{res}', block)

        if self.c_dim > 0:
            self.mapping = MappingNetwork(
                z_dim=0, c_dim=self.c_dim, w_dim=cmap_dim, num_latent=None,
                embed_features=None, layer_features=None, activation=activation,
                lr_multiplier=0.01
            )
        self.b4 = DiscriminatorEpilogue(
            channels[4], cmap_dim=cmap_dim, resolution=4,
            img_channels=self.img_channels, conv_clamp=conv_clamp,
            mbstd_group_size=stddev_group,
            mbstd_num_channels=1, activation=activation,
        )

    def forward(self, x, c=None, force_fp32=False):
        for res in self.block_resolutions:
            x = getattr(self, f'b{res}')(x, force_fp32=force_fp32)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        x = self.b4(x, cmap)
        return x
