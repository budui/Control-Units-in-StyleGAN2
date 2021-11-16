import math
import random
from typing import cast, Union, Sequence

import torch
import torch.nn as nn

import raydl
from models import MODEL
from models.StyleGAN2_mine import EqualFullyConnectedLayer, MinibatchStdLayer
from models.ada_ops import (
    upfirdn2d,
    conv2d_resample,
    bias_act,
    fma
)


class PixelNorm(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(PixelNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return x * (x.square().mean(dim=self.dim, keepdim=True) + self.eps).rsqrt()


class Blur(nn.Module):
    def __init__(self, resample_filter):
        super().__init__()
        self.register_buffer("kernel", upfirdn2d.setup_filter(resample_filter) * 4)


class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            demodulate=True,  # Apply weight demodulation?
            up=1,  # Integer upsampling factor.
            down=1,  # Integer downsampling factor.
            blur_kernel=None,
            padding=0,  # Padding with respect to the upsampled image.
    ):
        super(ModulatedConv2d, self).__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.up = int(up)
        self.down = int(down)
        self.demodulate = demodulate

        self.padding = padding

        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.blur = Blur(blur_kernel) if blur_kernel is not None else None
        self.modulation = EqualFullyConnectedLayer(style_dim, in_channel, bias_init=1)

        self.weight_gain = 1 / math.sqrt(in_channel * (kernel_size ** 2))

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise=None, flip_weight=None,
                fused_modconv=None):
        batch_size = style.shape[0]
        _, out_channel, in_channel, kh, kw = self.weight.shape
        if fused_modconv is None:
            fused_modconv = (not self.training) and (x.dtype == torch.float32 or int(x.shape[0]) == 1)

        style = self.modulation(style) * self.weight_gain

        weight = self.weight
        # pre-normalizing the style vector s and each row of the weight tensor w
        # before applying weight modulation and demodulation
        # from StyleGAN2-ADA. appendix.D.1: Mixed-precision training.
        if x.dtype == torch.float16 and self.demodulate:
            # During demodulation, each filter have in_channels * kh * kw elements,
            # which are multiplied by style and then squared to sum.
            # so, pre-normalize like this to make sure the sum is smaller than 1.
            # filter / filter.max() / num_of_elements(filter)

            # normalize conv weight by dividing the maximum of each filter(max: I x Kw x Kh)
            weight = self.weight * self.weight_gain / self.weight.norm(float("inf"), dim=[2, 3, 4], keepdim=True)
            # normalize style vector by dividing the maximum of each filter-level style
            style = style / style.norm(float("inf"), dim=-1, keepdim=True)

        w = None
        dcoefs = None

        if self.demodulate or fused_modconv:
            w = weight * style.view(batch_size, -1, in_channel, 1, 1)  # N x O x I x Kw x kh
        if self.demodulate:
            dcoefs = (w.square().sum(dim=[2, 3, 4]) + self.eps).rsqrt()
        if self.demodulate and fused_modconv:
            w = w * dcoefs.reshape(batch_size, -1, 1, 1, 1)  # N x O x 1 x 1 x 1

        # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
        flip_weight = (self.up == 1) if flip_weight is None else flip_weight  # slightly faster
        if not fused_modconv:
            x = x * style.to(x.dtype).reshape(batch_size, -1, 1, 1)
            x = conv2d_resample.conv2d_resample(
                x=x, w=weight[0].to(x.dtype), f=self.blur.kernel / 4 if self.blur is not None else None, up=self.up,
                down=self.down,
                padding=self.padding, flip_weight=flip_weight,
            )
            if self.demodulate and noise is not None:
                x = fma.fma(x, dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1), noise.to(x.dtype))
            elif self.demodulate:
                x = x * dcoefs.to(x.dtype).reshape(batch_size, -1, 1, 1)
            elif noise is not None:
                x = x.add_(noise.to(x.dtype))
            return x

        batch_size = int(batch_size)
        x = x.reshape(1, -1, *x.shape[2:])
        w = w.reshape(-1, in_channel, kh, kw)
        x = conv2d_resample.conv2d_resample(
            x=x,
            w=w.to(x.dtype),
            f=self.blur.kernel / 4 if self.blur is not None else None,
            up=self.up,
            down=self.down,
            padding=self.padding,
            groups=batch_size,
            flip_weight=flip_weight,
        )
        x = x.reshape(batch_size, -1, *x.shape[2:])
        if noise is not None:
            x = x.add_(noise)
        return x


class FusedActivationFunc(nn.Module):
    def __init__(self, channel=None, bias=True, activation="linear", conv_clamp=None):
        super().__init__()
        assert not (channel is None and bias)
        self.bias = nn.Parameter(torch.zeros(channel)) if bias else None
        self.conv_clamp = conv_clamp
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.activation = activation

    def __repr__(self):
        return raydl.class_repr(self, ["activation", "conv_clamp"])

    def forward(self, x, gain=1, bias=None):
        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None

        assert bias is None or self.bias is None
        bias = self.bias if bias is None else bias
        bias = bias.squeeze().to(x.dtype) if bias is not None else None
        x = bias_act.bias_act(x, bias, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class NoiseInjection(nn.Module):
    def __init__(self, enable=True, up=1):
        super().__init__()
        self.enable = enable
        self.up = up
        if self.enable:
            self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if not self.enable:
            return None
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height * self.up, width * self.up).normal_()
        return self.weight.to(noise.dtype) * noise


class StyledConv(nn.Module):
    def __init__(
            self,
            in_channels,  # Number of input channels.
            out_channels,  # Number of output channels.
            style_dim,  # Intermediate latent (W) dimensionality.
            kernel_size=3,  # Convolution kernel size.
            up=1,  # Integer upsampling factor.
            use_noise=True,  # Enable noise input?
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            resample_filter=(1, 3, 3, 1),  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super(StyledConv, self).__init__()

        self.conv = ModulatedConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=True,
            up=up,
            blur_kernel=None if up == 1 else resample_filter,
            padding=kernel_size // 2,
        )
        self.noise = NoiseInjection(use_noise, up=up)
        self.activate = FusedActivationFunc(out_channels, bias=True, activation=activation, conv_clamp=conv_clamp)

    def forward(self, x, w, noise=None, gain=1):
        out = self.conv(x, w, noise=self.noise(x, noise))
        out = self.activate(out, gain=gain)
        return out


class Upsample(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.register_buffer("kernel", upfirdn2d.setup_filter(kernel) * 4)

    def forward(self, x):
        return upfirdn2d.upsample2d(x, self.kernel / 4)


class ToRGB(nn.Module):
    def __init__(
            self,
            style_dim,
            in_channel,
            out_channel=3,
            upsample=True,
            conv_clamp=None,
            kernel_size=1,
            blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()

        self.conv_clamp = conv_clamp

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size=kernel_size,
            style_dim=style_dim,
            demodulate=False,
            up=1,
        )
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        self.activate = FusedActivationFunc(conv_clamp=conv_clamp, bias=False, activation="linear")
        if upsample:
            self.upsample = Upsample(blur_kernel)

    def forward(self, x, w, skip=None):
        out = self.conv(x, w)
        out = self.activate(out, bias=self.bias)

        if skip is not None:
            skip = self.upsample(skip)
            out.to(dtype=torch.float32)
            out = out + skip
        return out


class ConstantInput(nn.Module):
    def __init__(self, channel, resolution: Union[int, Sequence] = 4):
        super().__init__()
        resolution = (resolution, resolution) if isinstance(resolution, int) else resolution
        self.input = nn.Parameter(torch.randn(1, channel, *resolution))

    def forward(self, x):
        batch = x.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        return out


@MODEL.register_module("StyleGAN2-Generator")
class Generator(nn.Module):
    def __init__(
            self,
            size,  # (height, width) or size
            style_dim=512,
            n_mlp=8,
            img_channels=3,
            channel_multiplier=2,
            blur_kernel=(1, 3, 3, 1),
            lr_mlp=0.01,
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            num_fp16_res=0,  # Use FP16 for the N highest resolutions.
    ):
        super().__init__()

        size = (size, size) if isinstance(size, (int, float)) else size
        self.size = size
        self.style_dim = style_dim
        self.channel_multiplier = channel_multiplier
        self.log_size = int(math.log(max(size), 2))
        # Each resolution use two latent code and resolution start from 2**2.
        self.n_latent = (self.log_size - 1) * 2
        self.num_layers = (self.log_size - 2) * 2 + 1

        mapping_layers = [PixelNorm()]
        for _ in range(n_mlp):
            mapping_layers.append(
                EqualFullyConnectedLayer(style_dim, style_dim, lr_multiplier=lr_mlp, activation="lrelu")
            )

        self.style = nn.Sequential(*mapping_layers)

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

        fp16_resolution = max(2 ** (self.log_size + 1 - num_fp16_res), 8)
        self.resolution_to_fp16_enabled = [4 >= fp16_resolution]

        self.input = ConstantInput(self.channels[4], resolution=[int(s / 2 ** (self.log_size - 2) + 0.5) for s in size])
        self.conv1 = StyledConv(
            self.channels[4],
            self.channels[4],
            style_dim=style_dim,
            kernel_size=3,
            resample_filter=blur_kernel,
            conv_clamp=conv_clamp,
        )
        self.to_rgb1 = ToRGB(
            style_dim,
            in_channel=self.channels[4],
            out_channel=img_channels,
            conv_clamp=conv_clamp,
            kernel_size=1,
            upsample=False,
        )

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = 2 ** ((layer_idx + 5) // 2)
            h_radio = self.size[0] / (2 ** self.log_size)
            w_radio = self.size[1] / (2 ** self.log_size)
            shape = [1, 1, int(res * h_radio), int(res * w_radio)]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.resolution_to_fp16_enabled.append(2 ** i >= fp16_resolution)
            assert isinstance(out_channel, int)
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    style_dim=style_dim,
                    kernel_size=3,
                    up=2,
                    resample_filter=blur_kernel,
                    conv_clamp=conv_clamp,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel,
                    out_channel,
                    style_dim=style_dim,
                    kernel_size=3,
                    resample_filter=blur_kernel,
                    conv_clamp=conv_clamp,
                )
            )
            self.to_rgbs.append(
                ToRGB(
                    style_dim,
                    in_channel=out_channel,
                    out_channel=img_channels,
                    kernel_size=1,
                    conv_clamp=conv_clamp,
                ))

            in_channel = out_channel

    @torch.no_grad()
    def mean_latent(self, n_latent=4096):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def forward(
            self,
            z,
            return_latents=False,
            inject_index=None,
            noise=None,
            randomize_noise=True,
            truncation=1,
            truncation_latent=None,
    ):
        styles = [self.style(z_) for z_ in z] if isinstance(z, (list, tuple)) else [self.style(z)]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        latent = latent.to(torch.float32)

        out = self.input(latent)

        # first resolution
        dtype = torch.float16 if self.resolution_to_fp16_enabled[0] else torch.float32
        out.to(dtype)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                cast(nn.ModuleList, self.convs[::2]),
                cast(nn.ModuleList, self.convs[1::2]),
                noise[1::2],
                noise[2::2],
                self.to_rgbs,
        ):
            dtype = torch.float16 if self.resolution_to_fp16_enabled[i // 2 + 1] else torch.float32
            out = out.to(dtype=dtype)
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        image = skip

        if return_latents:
            return image, latent
        return image, None


class Conv2dLayer(nn.Module):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=(1, 3, 3, 1),  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.up = int(up)
        self.down = int(down)
        self.conv_clamp = conv_clamp

        self.register_buffer('kernel', upfirdn2d.setup_filter(resample_filter))

        self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]))
        self.padding = kernel_size // 2

        self.weight_gain = 1 / math.sqrt(in_channels * (kernel_size ** 2))

        self.activate = FusedActivationFunc(out_channels, bias, activation=activation, conv_clamp=conv_clamp)

    def forward(self, x, gain=1):
        weight = self.weight.mul(self.weight_gain).to(x.dtype)

        # when up==1, flip weight for slightly faster.
        # flip weight change the operation from convolution to correlation
        # maybe something related to implementation details between tensorflow and pytorch?
        x = conv2d_resample.conv2d_resample(
            x=x, w=weight, f=self.kernel, up=self.up, down=self.down,
            padding=self.padding, flip_weight=self.up == 1
        )
        return self.activate(x, gain=gain)


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            activation='lrelu',
            resample_filter=(1, 3, 3, 1),  # Low-pass filter to apply when resampling activations.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            fp16_enabled=False,  # Use FP16 for this block?
    ):
        super(ResBlock, self).__init__()

        self.fp16_enabled = fp16_enabled

        self.conv1 = Conv2dLayer(in_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.conv2 = Conv2dLayer(in_channels, out_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp,
                                 down=2, resample_filter=resample_filter)
        self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, down=2,
                                resample_filter=resample_filter)

    def forward(self, x, force_fp32=False):
        # fp16 stuff
        dtype = torch.float16 if self.fp16_enabled and not force_fp32 else torch.float32
        x = x.to(dtype)

        gain = 0.5 ** 0.5
        skip = self.skip(x, gain=gain)
        out = self.conv2(self.conv1(x), gain=gain)
        out = skip.add_(out)

        assert out.dtype == dtype
        return out


@MODEL.register_module("StyleGAN2-Discriminator")
class Discriminator(nn.Module):
    def __init__(
            self,
            size,
            channel_multiplier=2,
            num_image_channel=3,
            blur_kernel=(1, 3, 3, 1),
            activation='lrelu',  # Activation function: 'relu', 'lrelu', etc.
            stddev_group=4,  # Group size for the minibatch standard deviation layer, None = entire minibatch.
            num_fp16_res=0,  # Use FP16 for the N highest resolutions.
            conv_clamp=None,  # Clamp the output of convolution layers to +-X, None = disable clamping.
    ):
        super(Discriminator, self).__init__()

        size = (size, size) if isinstance(size, (int, float)) else size

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

        self.fp16_enabled = num_fp16_res > 0

        self.log_size = int(math.log(max(size), 2))

        convs = [Conv2dLayer(num_image_channel, channels[2 ** self.log_size], kernel_size=1, conv_clamp=conv_clamp)]

        in_channel = channels[2 ** self.log_size]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(
                in_channel, out_channel, fp16_enabled=(self.log_size - i) < num_fp16_res,
                conv_clamp=conv_clamp, resample_filter=blur_kernel
            ))
            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        last_in_channel = channels[4]
        # last resolution
        resolution = [int(s / 2 ** (self.log_size - 2) + 0.5) for s in size]
        # Group size for the minibatch standard deviation layer, None = entire minibatch.
        self.stddev_group = stddev_group
        # Number of features for the minibatch standard deviation layer, 0 = disable.
        self.stddev_feat = 1

        self.mbstd = MinibatchStdLayer(group_size=self.stddev_group,
                                       num_channels=self.stddev_feat) if self.stddev_feat > 0 else None
        self.final_conv = Conv2dLayer(in_channel + self.stddev_feat, last_in_channel, kernel_size=3,
                                      activation=activation, conv_clamp=conv_clamp)
        self.final_linear = nn.Sequential(
            EqualFullyConnectedLayer(
                last_in_channel * (resolution[0] * resolution[1]), last_in_channel, activation=activation),
            EqualFullyConnectedLayer(last_in_channel, 1),
        )

    def forward(self, x):
        dtype = torch.float16 if self.fp16_enabled else torch.float32

        x = x.to(dtype)
        x = self.convs(x)

        dtype = torch.float32
        x = x.to(dtype)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.final_conv(x)
        x = self.final_linear(x.flatten(1))
        return x
