from pathlib import Path

import fire
import torch
from loguru import logger

import editing.util
import raydl
from editing.config import CHECKPOINT_PATH, STATISTICS_PATH, CHANNEL_MAP, CORRECTION_PATH
from editing.util import load_latent, channel_selector
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator

__all__ = ["Mixer", "Manipulator", "ManipulatorBuilder"]


class Mixer:
    def __init__(self, mix_weight=0.5, select=None):
        self.data = select
        self.mix_weight = mix_weight

    @staticmethod
    def load(mixer_str):
        if isinstance(mixer_str, (int, float)) or "#" not in mixer_str:
            return Mixer(mix_weight=float(mixer_str))
        path, weight = mixer_str.split("#")
        data = torch.load(path)
        m = Mixer(float(weight), data)
        return m

    def apply(self, styles1, styles2):
        mix_weight = self.mix_weight
        if self.data is None:
            return [s1 * mix_weight + s2 * (1 - mix_weight) for s1, s2 in zip(styles1, styles2)]
        mix_style = []
        for i in range(len(styles1)):
            if i in self.data:
                self.data[i].to(styles1[i].device)
                mix_style.append(styles1[i] + mix_weight * self.data[i] * (styles2[i] - styles1[i]))
            else:
                mix_style.append(styles1[i] * mix_weight + styles2[i] * (1 - mix_weight))
        return mix_style


class Manipulator:
    """
    Responsible for moving or replacing latent codes.
    """

    def __init__(self, data=None, channel_multiplier=2, device=torch.device("cpu")):
        self.data = data or {}
        self.channel_multiplier = channel_multiplier
        self.device = device
        self.to(self.device)

    @staticmethod
    def _num_channels(layer, channel_multiplier=2):
        # resolution = 2**log_size
        log_size = (layer + 1) // 3 + 2

        channels = {
            2: 512,
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

        if layer % 3 == 1:
            # to_rgb layer
            return channels[int(2 ** log_size)], 3
        if layer % 3 == 0:
            return channels[int(2 ** log_size)], channels[int(2 ** log_size)]
        else:
            return channels[int(2 ** (log_size - 1))], channels[int(2 ** log_size)]

    def _init_alpha(self, layer, alpha_indices=None):
        num_output_channel = self._num_channels(layer, self.channel_multiplier)[1]
        alpha = torch.zeros(num_output_channel, device=self.device)
        if alpha_indices is None:
            alpha.fill_(1)
        else:
            alpha[alpha_indices] = 1
        return alpha

    def replacements(self):
        return {layer: tool for layer, tool in self.data.items() if "replace" in tool}

    def movements(self):
        return {layer: tool for layer, tool in self.data.items() if "move" in tool}

    def add_replacement(self, layer, style, replace_indexes=None):
        layer = int(layer)
        style = style.detach().squeeze().requires_grad_(False)
        assert style.dim() == 1 and len(style) == self._num_channels(layer, self.channel_multiplier)[0]

        alpha = self._init_alpha(layer, replace_indexes)

        if layer in self.data:
            self.data[layer]["replace"] = dict(style=style, alpha=alpha)
        else:
            self.data[layer] = dict(replace=dict(style=style, alpha=alpha))

    def add_movement(self, layer, direction, alpha=None, need_normalization=False, move_indices=None):
        layer = int(layer)
        direction = direction.detach().squeeze().requires_grad_(False).to(self.device)
        if need_normalization:
            direction = direction / direction.norm(2)

        if alpha is None:
            alpha = self._init_alpha(layer, alpha_indices=move_indices)

        if layer in self.data:
            self.data[layer]["move"] = dict(direction=direction, alpha=alpha)
        else:
            self.data[layer] = dict(move=dict(direction=direction, alpha=alpha))

    def save(self, save_path):
        save_path = Path(save_path)
        if not save_path.parent.exists():
            save_path.parent.mkdir()
        torch.save(self.data, save_path)

    @staticmethod
    def load(mdfc, device=torch.device("cpu")):
        if isinstance(mdfc, Manipulator):
            return mdfc.to(device)
        if isinstance(mdfc, (str, Path)):
            data = torch.load(mdfc)
            m = Manipulator(data)
            m.to(device)
            return m
        raise ValueError(f"invalid mdfc, expect path to mdfc or Manipulator, but got {mdfc}")

    def to(self, device=torch.device("cpu")):
        for layer in self.data:
            for t in self.data[layer]:
                for k in self.data[layer][t]:
                    if not torch.is_tensor(self.data[layer][t][k]):
                        self.data[layer][t][k] = torch.tensor(self.data[layer][t][k])
                    self.data[layer][t][k] = self.data[layer][t][k].to(device, torch.float)
        self.device = device
        return self

    def apply(self, styles, move_factor=None, replace_factor=None, discard_alpha=False):
        """
        apply modification to `styles`.
        :param styles:
        :param move_factor:
        :param replace_factor:
        :param discard_alpha:
        :return:
        """
        if len(self.data) == 0:
            return styles

        for layer in self.data:
            batch_size = styles[layer].size(0)

            style = styles[layer]
            if style.dim() == 2:
                style = style.unsqueeze(dim=1)

            if "move" in self.data[layer]:
                direction = self.data[layer]["move"]["direction"].to(style)
                alpha = self.data[layer]["move"]["alpha"].to(style)
                if move_factor is None:
                    # Randomly sample a move_factor from predefined range to see manipulation result.
                    typical_range = self.data[layer]["move"].get("typical_range", None)
                    assert typical_range is not None
                    typical_min, typical_max = typical_range
                    move_factor = (2 + torch.rand(1)) * (typical_max - typical_min) / 3 + typical_min
                    logger.info(
                        f"random sample a move_factor {move_factor} from [{typical_min}, {typical_max})"
                        f"because have not got a move_factor"
                    )
                if not torch.is_tensor(move_factor):
                    move_factor = style.new_tensor(move_factor)
                move_factor = move_factor.expand(style.size(0))
                if discard_alpha:
                    alpha = torch.ones_like(alpha)
                style = style.expand(batch_size, alpha.size(-1), style.size(-1))
                # 1xOxI
                direction = direction.view(1, 1, -1).expand(1, alpha.size(-1), -1) * alpha.view(1, -1, 1)
                style = style + move_factor.view(style.size(0), 1, 1) * direction

            if "replace" in self.data[layer]:
                if replace_factor is None:
                    replace_factor = self.data[layer]["replace"].get("typical_replace_factor", None)
                    assert replace_factor is not None
                    logger.info(f"random sample a replace_factor {replace_factor}")
                target_style = self.data[layer]["replace"]["style"].to(style)
                alpha = self.data[layer]["replace"]["alpha"].to(style)
                style = style.expand(batch_size, alpha.size(-1), style.size(-1))
                # 1xOxI
                target_style = target_style.view(1, 1, -1).expand(1, alpha.size(-1), -1)
                style = style + replace_factor * alpha.view(1, -1, 1) * (target_style - style)
            styles[layer] = style
        return styles

    def __repr__(self):
        repr_str = "Manipulator(\n"
        for layer in self.data:
            if "move" in self.data[layer]:
                direction = self.data[layer]["move"]["direction"]
                alpha = self.data[layer]["move"]["alpha"]
                repr_str += f"layer_{layer}: direction(norm={direction.norm():.4f}, " \
                            f"nonzero={len(torch.nonzero(direction))}, alpha(sum={alpha.sum()})"

                if "typical_range" in self.data[layer]["move"]:
                    repr_str += f"\t typical_range:{self.data[layer]['move']['typical_range']}"

            if "replace" in self.data[layer]:
                non_zero_alpha = torch.nonzero(self.data[layer]['replace']['alpha']).flatten(0)
                repr_str += f"layer_{layer}: " \
                            f"replace(number={len(non_zero_alpha)}, example:" \
                            f"{non_zero_alpha if len(non_zero_alpha) < 16 else str(non_zero_alpha[:16]) + '...'})"
                if "typical_replace_factor" in self.data[layer]["replace"]:
                    repr_str += f"\t typical_replace_factor: " \
                                f"{self.data[layer]['replace']['typical_replace_factor']}"
            repr_str += "\n"
        repr_str += ")"
        return repr_str


class ManipulatorBuilder:
    """
    each method of this class can be used to create new mdfc.
    """

    def __init__(self, save=True, checkpoint=CHECKPOINT_PATH, save_folder="./tmp", generator=None, mdfc_name=None,
                 device=torch.device("cuda")):
        self.save = save
        self.save_folder = Path(save_folder)
        self.device = device
        self._G = generator
        self.mdfc_name = mdfc_name
        self.checkpoint = checkpoint

    @property
    def G(self):
        if self._G is None:
            self._G = ImprovedStyleGAN2Generator.load(self.checkpoint, device=self.device)
            self._G.manipulation_mode()
            self._G.eval()
        return self._G

    def _postprocess(self, mdfc: Manipulator, save_name):
        if self.save:
            if not self.save_folder.exists():
                self.save_folder.mkdir()
            save_path = self.save_folder / f"{save_name}.mdfc"
            mdfc.save(save_path)
            return str(save_path)
        else:
            return mdfc, save_name

    def _styles(self, latent_dict):
        with torch.no_grad():
            if "styles" in latent_dict:
                styles = latent_dict["styles"]
            elif "w" in latent_dict:
                styles = self.G.w_to_styles(latent_dict["w"])
            elif "z" in latent_dict:
                styles = self.G.w_to_styles(self.G.z_to_w(latent_dict["z"]))
        return styles

    def single_channel(self, layers, indices):
        """
        use (layer,index) pairs to specific style channels.
        """
        statistics = torch.load(STATISTICS_PATH)
        for layer, index in zip(*raydl.paired_indexes(layers, indices)):
            mdfc = Manipulator()
            direction = torch.zeros(CHANNEL_MAP[layer])
            direction[index] = statistics["std_styles"][layer][0, index]
            mdfc.add_movement(layer, direction)

            yield self._postprocess(
                mdfc,
                f"single_channel_{layer}_{index}" if self.mdfc_name is None else f"{self.mdfc_name}_{layer}_{index}"
            )

    def alter(self, mdfc_path, layers, rules="all", is_indexes_rule=False, correction_path=CORRECTION_PATH,
              alter_type="move_direction", alter_rate=1):
        """
        only KEEP the channels selected by rules
        """
        mdfc = Manipulator.load(mdfc_path, device=self.device)
        mdfc_path = Path(mdfc_path) if isinstance(mdfc_path, (str, Path)) else self.save_folder / "tmp.mdfc"

        masks = channel_selector(layers, rules, is_indexes_rule, correction_path)

        m = Manipulator()

        alter_k, alter_v = alter_type.split("_")
        for layer in mdfc.data:
            layer = layer if alter_type == "move_direction" else editing.util.next_layer(layer)
            if layer in masks:
                logger.debug(f"layer_{layer}: norm({alter_type}): {mdfc.data[layer][alter_k][alter_v].norm(2)}")
                mdfc.data[layer][alter_k][alter_v] *= masks[layer].to(self.device).squeeze() * alter_rate
                logger.debug(f"  after_alter: norm({alter_type}): {mdfc.data[layer][alter_k][alter_v].norm(2)}")

            if "move" in mdfc.data[layer] and layer in masks:
                if layer not in m.data:
                    m.data[layer] = dict(move=mdfc.data[layer]["move"])
                else:
                    m.data[layer]["move"] = mdfc.data[layer]["move"]
            if "replace" in mdfc.data[layer]:
                if layer not in m.data:
                    m.data[layer] = dict(replace=mdfc.data[layer]["replace"])
                else:
                    m.data[layer]["replace"] = mdfc.data[layer]["replace"]

        yield self._postprocess(
            m,
            mdfc_path.parent / (f"alter_{mdfc_path.stem}" if self.mdfc_name is None else self.mdfc_name)
        )

    def remove(self, mdfc_path, layers, rules="all", is_indexes_rule=False, correction_path=CORRECTION_PATH,
               alter_type="move_direction", alter_rate=1):
        """
        only REMOVE the channels selected by rules
        """
        mdfc_path = Path(mdfc_path)
        mdfc = Manipulator.load(mdfc_path, device=self.device)

        masks = channel_selector(layers, rules, is_indexes_rule, correction_path)

        alter_k, alter_v = alter_type.split("_")
        for layer in mdfc.data:
            layer = layer if alter_type == "move_direction" else editing.util.next_layer(layer)
            if layer in masks:
                logger.debug(f"layer_{layer}: norm({alter_type}): {mdfc.data[layer][alter_k][alter_v].norm(2)}")
                mdfc.data[layer][alter_k][alter_v] *= 1 - masks[layer].to(self.device).squeeze() * alter_rate
                logger.debug(f"  after_alter: norm({alter_type}): {mdfc.data[layer][alter_k][alter_v].norm(2)}")
        yield self._postprocess(
            mdfc,
            mdfc_path.parent / (f"remove_{mdfc_path.stem}" if self.mdfc_name is None else self.mdfc_name)
        )

    def replace(self, latent_family, layers, family_ids=None, channel_multiplier=2):
        """
        replace the selected `layers` with latent code in `latent_family`
        """
        if self._G is not None:
            channel_multiplier = self._G.channel_multiplier
        base_save_name = self.mdfc_name or f"replace_{Path(latent_family).stem}"
        latent_family = load_latent(latent_family, ids=family_ids, load_real=False, latent_type=None,
                                    device=self.device)

        styles = self._styles(latent_family)

        for i in range(len(styles[0])):
            m = Manipulator(channel_multiplier=channel_multiplier)
            for l in raydl.tuple_of_indices(layers):
                m.add_replacement(l, style=styles[l][i:i + 1])
            m.to(self.device)

            yield self._postprocess(m, f"{base_save_name}_{i}")

    def paired_delta(self, latent1, latent2, ids1=None, ids2=None, layers=None, rules="all",
                     correction_path=CORRECTION_PATH):
        """
        use the difference of the mean of two class of latents as modification
        """
        latent_dict1 = load_latent(latent1, ids=ids1, device=self.device)
        latent_dict2 = load_latent(latent2, ids=ids2, device=self.device)

        styles1 = self._styles(latent_dict1)
        styles2 = self._styles(latent_dict2)
        m = Manipulator()
        for l, (s1, s2) in enumerate(zip(styles1, styles2)):
            m.add_movement(l, s1.mean(dim=0, keepdim=True) - s2.mean(dim=0, keepdim=True))

        save_name = f"paired_delta_{Path(latent1).stem}_{ids1}_{Path(latent2).stem}_{ids2}"
        if layers is None:
            yield self._postprocess(m, save_name)
        else:
            old_save = self.save
            self.save = False
            m, _ = next(self.alter(m, layers, rules, correction_path=correction_path))
            self.save = old_save
            yield self._postprocess(m, f"alter_{save_name}")

    def sefa(self, eigvec_ids, eigvec_layers=None, layers=None):
        eigvec_ids = raydl.tuple_of_indices(eigvec_ids)
        layers = raydl.tuple_of_indices(layers) if layers is not None else None
        eigvec_layers = raydl.tuple_of_indices(eigvec_layers) if eigvec_layers is not None \
            else sum([[3 * i - 1, 3 * i] for i in range(1, 9)], [0, 1])

        modulate = {
            k: v
            for k, v in self.G.state_dict().items()
            if "modulation" in k and "weight" in k
        }
        weights = torch.cat([w for i, w in enumerate(modulate.values()) if i in eigvec_layers])
        eigvec = torch.svd(weights).V.to(self.device)  # [512x512]
        for idx in eigvec_ids:
            m = Manipulator()

            w_vector = eigvec[:, idx].unsqueeze(0)
            w_vector /= w_vector.norm(2)
            delta_styles = self.G.w_to_styles(w_vector)
            zero_styles = self.G.w_to_styles(torch.zeros_like(w_vector))

            for layer, zero, delta in zip(range(len(zero_styles)), zero_styles, delta_styles):
                if layers is None or layer in layers:
                    m.add_movement(layer, delta - zero)

            yield self._postprocess(m, f"sefa_{idx}")

    def ganspace(self, principal_direction_ids, layers=None, styles_layers=None):
        principal_direction_ids = raydl.tuple_of_indices(principal_direction_ids)
        layers = raydl.tuple_of_indices(layers) if layers is not None else None
        styles_layers = raydl.tuple_of_indices(styles_layers) if styles_layers is not None else None

        ws = []
        for batch in raydl.total_chunk(300_000, 4096):
            z = torch.randn(batch, 512, device=self.device)
            ws.append(self.G.z_to_w(z))
        ws = torch.cat(ws)

        u, s, v = torch.pca_lowrank(ws, q=max(principal_direction_ids) + 1)

        for idx in principal_direction_ids:
            m = Manipulator()

            w_vector = v[:, idx].unsqueeze(0)
            w_vector /= w_vector.norm(2)
            delta_styles = self.G.w_to_styles(w_vector)
            zero_styles = self.G.w_to_styles(torch.zeros_like(w_vector))

            for layer, zero, delta in zip(range(len(zero_styles)), zero_styles, delta_styles):
                if layers is None or layer in layers:
                    m.add_movement(layer, delta - zero)

            yield self._postprocess(m, f"ganspace_w_{idx}")


if __name__ == '__main__':
    fire.Fire(ManipulatorBuilder)
