from pathlib import Path

import fire
import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger
from tqdm import tqdm

import raydl
from editing import util
from editing.config import CHECKPOINT_PATH, FACE_PARSER_CKP
from editing.modification import Manipulator
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator
from third_party.BiSetNet import FaceParser


def crop_vector(vector, rate=0.8, exponent=2):
    assert 1 >= rate >= 0
    vector = vector.flatten().abs()
    vector_norm_sum = vector.pow(exponent).sum()
    min_norm = vector_norm_sum * rate
    values, indexes = vector.pow(exponent).topk(len(vector))
    norm = 0
    i = 0
    while norm <= min_norm:
        norm += values[i]
        i += 1
    main_indexes = indexes[:i]
    mask = torch.zeros_like(vector)
    mask[main_indexes] = 1
    return vector * mask


def prev(
        modification_path,
        mask_ids,
        checkpoint=CHECKPOINT_PATH,
        num_batch=500,
        batch_size=16,
        save_path=None,
        device="cuda",
        seed=None,
        lr=0.1,
        delta_s_min_rate=0.6,
        min_resolution=32,
        lambda_neg=1,
        replace_factor=1,
        compact_mask=True,
        truncation=0.7,
):
    print(locals())
    G = ImprovedStyleGAN2Generator.load(checkpoint, device=device, default_truncation=truncation)
    G.manipulation_mode()
    G.eval()
    G.requires_grad_(False)

    if seed is not None:
        torch.manual_seed(seed)

    mask_ids = raydl.tuple_of_indices(mask_ids)

    face_parser = FaceParser(model_path=FACE_PARSER_CKP)

    if save_path is None:
        save_path = Path(modification_path).parent / f"before_{Path(modification_path).stem}.mdfc"
    else:
        save_path = Path(save_path)

    modification = Manipulator.load(modification_path, device)

    with torch.no_grad():
        w = G.z_to_w(torch.randn(4096, G.style_dim, device=device), truncation=truncation)
        std_styles = [s.std(dim=0, keepdim=True) for s in G.w_to_styles(w)]
        mean_styles = G.w_to_styles(w.mean(dim=0, keepdim=True))

    min_layer = min(modification.data.keys())
    prev_layer = util.prev_layer(min_layer)

    delta_s_indices = torch.flatten(torch.nonzero(crop_vector(
        vector=modification.data[min_layer]["move"]["direction"] / std_styles[min_layer].to(device),
        rate=delta_s_min_rate,
    ))).tolist()
    logger.info(f"optimize for {len(delta_s_indices)} dims: {delta_s_indices}")

    assert "move" in modification.data[min_layer]

    modification.add_replacement(prev_layer, mean_styles[prev_layer], replace_indexes=delta_s_indices)
    modification.to(device)
    modification.data[prev_layer]["replace"]["style"].requires_grad_(True)

    optimizer = optim.Adam([modification.data[prev_layer]["replace"]["style"]], lr=lr)

    for i in tqdm(range(num_batch), ncols=0):
        z = torch.randn(batch_size, 512, device=device)
        w = G.z_to_w(z)
        zero_styles = G.w_to_styles(w)
        zero_image = G.styles_to_image(zero_styles)

        _, feature = G.styles_to_image_and_features(
            modification.apply(styles=zero_styles, move_factor=0, replace_factor=replace_factor),
            min_layer,
        )
        feature = feature[0]

        with torch.no_grad():
            parsing = face_parser.batch_run(zero_image, pre_normalize=True, image_repr=False, compact_mask=compact_mask)
            if parsing is None:
                continue
            mask = torch.zeros(parsing.size(0), 1, *zero_image.size()[-2:]).to(device)
            for mi in mask_ids:
                mask = torch.logical_or(mask, parsing[:, mi: mi + 1])
            mask = mask.float()

        if feature.size(-1) < min_resolution:
            feature = F.interpolate(feature, min_resolution)
            mask = F.interpolate(mask, min_resolution)
        else:
            mask = F.interpolate(mask, feature.size()[-2:])

        mask_factor = mask.sum() / torch.numel(mask)

        target_feature = feature[:, delta_s_indices].abs()
        feature_pos_loss = -(mask * target_feature).mean() / mask_factor
        feature_neg_loss = ((1 - mask) * target_feature).mean() / (1 - mask_factor)

        loss = feature_pos_loss + lambda_neg * feature_neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"total: {loss.item():.2f} disentangle: {feature_neg_loss.item():.2f} "
                  f"effect: {feature_pos_loss.item():.2f} ")

    modification.save(save_path)


if __name__ == '__main__':
    fire.Fire()
