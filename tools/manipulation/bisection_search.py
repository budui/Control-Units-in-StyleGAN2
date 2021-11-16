from pathlib import Path
from shutil import copyfile

import fire
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from loguru import logger
from torchvision.transforms.functional import to_tensor, normalize
from tqdm import tqdm

import raydl
from editing.config import RECORD_PATH, CLASSIFIER_CKP, FACE_PARSER_CKP, CHECKPOINT_PATH
from editing.modification import Manipulator
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator
from third_party.BiSetNet import FaceParser
from third_party.attributes_classifier import AttrClassifier

VALID_ATTR_IDS = sorted(list(set(range(40)) - {10, 23, 37, 25, 29, 27, 4, 22, 26}))


# ATTR_RATE = [0.2857, 0.2184, 0.2672, 0.7640, 0.0089, 0.1105, 0.5377, 0.7902, 0.2599,
#              0.0816, 0.0048, 0.2198, 0.1524, 0.2775, 0.1891, 0.1535, 0.1135, 0.0502,
#              0.0970, 0.5659, 0.4730, 0.6362, 0.0819, 0.2780, 0.6954, 0.0893, 0.0588,
#              0.0140, 0.1115, 0.0199, 0.1790, 0.6356, 0.1937, 0.1983, 0.4970, 0.0462,
#              0.2399, 0.3196, 0.0789, 0.7150]


class BisectWorker(object):
    def __init__(self, record_path=RECORD_PATH, device=torch.device("cuda"), save_path="tmp"):
        self.device = device
        records = np.load(record_path)
        # 500000x40
        self.logit_bank = torch.from_numpy(records["logit"]).to(device, torch.float)
        # 500000x512
        self.w_bank = torch.from_numpy(records["w"]).to(device, torch.float)
        # 40
        self.std_bank = self.logit_bank.std(dim=0)

        self.save_path = Path(save_path)
        if not self.save_path.exists():
            self.save_path.mkdir(parents=True)

        self.classifier = AttrClassifier(CLASSIFIER_CKP, device=self.device)
        self.face_parser = FaceParser(model_path=FACE_PARSER_CKP)

        self.G = ImprovedStyleGAN2Generator.load(CHECKPOINT_PATH, device=self.device)
        self.G.manipulation_mode()
        self.G.eval()

    def _synthesize(self, w=None, styles=None, modifications=None):
        styles = styles or self.G.w_to_styles(w)
        if modifications is not None:
            assert isinstance(modifications, (list, tuple))
            for mdfc, apply_param in modifications:
                styles = mdfc.apply(styles, **apply_param)
        image = self.G.styles_to_image(styles)
        return image

    def _infer(self, image, use_logit=True):
        return self.classifier.infer(image, resize=224, logit=use_logit)[0]

    @torch.no_grad()
    def _image_attr(self, mdfc=None, replace_factor=1, move_factors=None, w=None, styles=None, use_logit=True):
        ars = []
        for factor in move_factors:
            modifications = (
                [(mdfc, dict(replace_factor=replace_factor, move_factor=factor))] if mdfc is not None else None
            )
            image = self._synthesize(w=w, styles=styles, modifications=modifications)
            ars.append(self._infer(image, use_logit=use_logit))
        return torch.stack(ars)

    def _real_target(self, attr_id, target=None, delta=None, use_logit=True, zero_attr=None):
        if delta is not None:
            assert zero_attr is not None
            if use_logit:
                return zero_attr[attr_id] + self.std_bank[attr_id] * delta
            return zero_attr[attr_id] + delta
        if target is not None:
            return target
        raise ValueError

    def grid_search(self, attr_id, w, mdfc, target, start_factor, end_factor, num_factors, replace_factor, use_logit):
        """
        must ensure logit(start_factor) < logit(end_factor)
        """
        move_factors = raydl.factors_sequence(end_factor=end_factor, start_factor=start_factor, num_factors=num_factors)
        move_factors = torch.FloatTensor(move_factors).to(self.device)
        attrs = self._image_attr(
            mdfc, replace_factor=replace_factor, move_factors=move_factors, w=w, use_logit=use_logit
        )

        start_factor = torch.masked_select(move_factors, attrs[:, attr_id].le(target))
        end_factor = torch.masked_select(move_factors, attrs[:, attr_id].ge(target))

        logger.debug(
            f"\ntarget: {target}\n"
            + "\t".join(map(lambda x: f"{x:.2f}", move_factors.tolist()))
            + "\n"
            + "\t".join(map(lambda x: f"{x:.2f}", attrs[:, attr_id].tolist()))
        )

        if len(end_factor) == 0 or len(start_factor) == 0:
            return None, None

        end_factor = end_factor[end_factor.abs().argmin()]
        start_factor = move_factors[move_factors.tolist().index(end_factor) - 1]
        return start_factor, end_factor

    def pair_evaluate(self, attr_id, original_image, current_image, original_attr, current_attr, mask_ids,
                      original_mask=True):
        mse_error = F.mse_loss(original_image, current_image)

        parsing = self.face_parser.batch_run(
            original_image if original_mask else current_image,
            pre_normalize=True,
            image_repr=False,
            compact_mask=True
        )
        if parsing is None:
            mse_valid = None
        else:
            mask = torch.zeros(parsing.size(0), 1, *original_image.size()[-2:]).to(self.device)
            for mi in mask_ids:
                mask = torch.logical_or(mask, parsing[:, mi: mi + 1])
            mask = mask.float()
            mse_valid = F.mse_loss(original_image * mask, current_image * mask) / mse_error

        compared_attr_ids = list(set(VALID_ATTR_IDS) - {attr_id, })
        ad = ((original_attr - current_attr).abs() / self.std_bank)[compared_attr_ids]
        mean_ad = ad.mean()
        max_ad = ad.max()
        return mse_error, mse_valid, mean_ad, max_ad

    def load_image(self, image_path, resize=None):
        img = to_tensor(Image.open(image_path))
        img = img.unsqueeze_(dim=0).to(self.device)
        img = normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if resize is not None:
            img = F.interpolate(img, resize)
        return img

    def search(
            self,
            attr_id,
            w,
            mdfc,
            start_factor,
            end_factor,
            replace_factor,
            num_grid_factors=5,
            tolerance=0.05,
            target=None,
            delta=None,
            use_logit=True,
            max_iteration=20,
            real_path=None,
    ):
        tolerance = tolerance * self.std_bank[attr_id] if use_logit else tolerance

        _zero_image = self._synthesize(w=w)
        zero_image = _zero_image if real_path is None else self.load_image(real_path)
        zero_attr = self._infer(zero_image, use_logit)
        target = self._real_target(attr_id, target=target, delta=delta, use_logit=use_logit, zero_attr=zero_attr)

        start_factor, end_factor = self.grid_search(
            attr_id, w, mdfc, target, start_factor, end_factor, num_grid_factors, replace_factor, use_logit
        )
        if start_factor is None:
            return "failed/grid_search", None, None, zero_image, None, zero_attr, _zero_image

        logger.info(f"start search between factor {start_factor} and {end_factor}")

        cur_factor = (start_factor + end_factor) / 2

        for _ in range(max_iteration):
            image = self._synthesize(
                w=w, modifications=[(mdfc, dict(replace_factor=replace_factor, move_factor=cur_factor))]
            )
            cur_attr = self._infer(image, use_logit)
            logger.debug(f"cur_factor: {cur_factor} attr: {cur_attr[attr_id].item():.2f}")
            if (cur_attr[attr_id] - target).abs() < tolerance:
                return "ok", cur_factor, image, zero_image, cur_attr, zero_attr, _zero_image
            elif cur_attr[attr_id] > target:
                end_factor = cur_factor
                if cur_factor == start_factor:
                    raise RuntimeError("bisection failed")
                cur_factor = (start_factor + cur_factor) / 2
            else:
                start_factor = cur_factor
                if cur_factor == end_factor:
                    raise RuntimeError("bisection failed")
                cur_factor = (end_factor + cur_factor) / 2

        return "failed/too_many_iterations", None, None, zero_image, None, zero_attr, _zero_image

    @torch.no_grad()
    def generate(self, w_path):
        w_path = Path(w_path)
        w_pool = torch.load(w_path).to(self.device)

        original_path = self.save_path / "original"
        if not original_path.exists():
            original_path.mkdir()

        for i in tqdm(range(len(w_pool))):
            w = w_pool[i: i + 1]
            zero_image = self._synthesize(w=w)
            raydl.save_images(zero_image, original_path / f"{i:06d}.jpg", resize=512, nrow=1)

    @torch.no_grad()
    def run(self, attr_id, mdfc, w_path, start_factor, end_factor, replace_factor, mask_ids,
            save_original=False, original_mask=True, num_samples=None, **search_param):
        mdfc = Manipulator.load(mdfc, self.device)

        w_path = Path(w_path)
        w_pool = torch.load(w_path).to(self.device)
        if num_samples is None:
            num_samples = len(w_pool)
        w_pool = w_pool[:num_samples]
        real_paths = None
        if w_path.with_suffix(".realpath").exists():
            logger.info("find real images! use real images to search!")
            real_paths = torch.load(w_path.with_suffix(".realpath"))
            real_paths = real_paths[:num_samples]
            assert len(real_paths) == len(w_pool)

        result_pool = []
        w_not_succ_pool = []

        current_path = self.save_path / "current"
        original_path = self.save_path / "original"
        real_folder = self.save_path / "real"

        if not current_path.exists():
            current_path.mkdir()
        if save_original and not original_path.exists():
            original_path.mkdir()
            real_folder.mkdir()

        for i in range(len(w_pool)):
            w = w_pool[i: i + 1]
            real_path = None if real_paths is None else Path(real_paths[i])

            result, factor, image, zero_image, cur_attr, zero_attr, _zero_image = self.search(
                attr_id, w, mdfc, start_factor, end_factor, replace_factor, real_path=real_path, **search_param
            )
            logger.info(f"index_{i}: {result} factor:{factor}")
            if result == "ok":
                result_pool.append((
                    i, factor,
                    *self.pair_evaluate(
                        attr_id, zero_image, image, zero_attr, cur_attr, mask_ids=raydl.tuple_of_indices(mask_ids),
                        original_mask=original_mask
                    )
                ))
                raydl.save_images(image, current_path / f"{i:06d}.jpg", resize=512, nrow=1)
            else:
                w_not_succ_pool.append(w)

            if save_original:
                raydl.save_images(_zero_image, original_path / f"{i:06d}.jpg", resize=512, nrow=1)
                if real_path is not None:
                    copyfile(real_path, real_folder / f"{i:06d}.jpg")

        if len(w_not_succ_pool) > 0:
            w_not_succ_pool = torch.cat(w_not_succ_pool)
            torch.save(w_not_succ_pool, self.save_path / f"{w_path.stem}_not_succ.w")
            bad_images = self.G.styles_to_image(self.G.w_to_styles(w_not_succ_pool[:8]))
            raydl.save_images(bad_images, self.save_path / f"failed_example.jpg", resize=256, nrow=8)

        mean_metrics = [torch.FloatTensor([r[i] for r in result_pool if r[i] is not None]).mean() for i in
                        range(2, len(result_pool[0]))]

        factors = torch.FloatTensor([r[1] for r in result_pool])
        cov = factors.std() / factors.mean()

        logger.info(f"coefficient of variation: {cov}")

        logger.info("\n" +
                    "\t".join(["mse_error", "mse_valid", "mean_ad", "max_ad"]) +
                    "\n" +
                    "\t".join(map(lambda x: f"{x:.4f}", mean_metrics)))

        logger.info(f"total: {len(w_pool)} succ: {len(result_pool)}, not_succ: {len(w_not_succ_pool)} "
                    f"succ_rate:{len(result_pool) / len(w_pool):.4f}")

        torch.save(
            dict(metrics=mean_metrics, pool=result_pool, rate_succ=len(result_pool) / len(w_pool), cov=cov),
            self.save_path / "result.pt"
        )


if __name__ == "__main__":
    fire.Fire(BisectWorker)
