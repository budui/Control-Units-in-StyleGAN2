import pickle
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import ignite.distributed as idist
import numpy as np
import torch
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce
from torch.hub import download_url_to_file

STYLEGAN2_ADA_FID_WEIGHTS_URL = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/' \
                                'pretrained/metrics/inception-2015-12-05.pt'


def fid_score_(sample_mean, sample_cov, real_mean, real_conv, eps=1e-6):
    try:
        import scipy
        import scipy.linalg
    except ImportError:
        raise RuntimeError("fid_score requires scipy to be installed.")
    m = np.square(sample_mean - real_mean).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sample_cov, real_conv), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sample_cov + real_conv - s * 2))
    return float(fid)


def fid_score(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    """Refer to the implementation from:

    https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
    """
    try:
        import scipy
        import scipy.linalg
    except ImportError:
        raise RuntimeError("fid_score requires scipy to be installed.")

    cov_sqrt, _ = scipy.linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = scipy.linalg.sqrtm(
            (sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return float(fid)


class StyleGAN2InceptionExtractor:
    def __init__(self, inception_path="./pretrained_models/stylegan2-ada-fid-inception.pt") -> None:
        self.device = idist.device()
        if not Path(inception_path).exists():
            if idist.get_local_rank() > 0:
                # Ensure that only local rank 0 download the checkpoint
                # Thus each node will download a copy of the checkpoint
                idist.barrier()
            if not Path(inception_path).parent.exists():
                Path(inception_path).parent.mkdir()
            download_url_to_file(STYLEGAN2_ADA_FID_WEIGHTS_URL, inception_path)
            if idist.get_local_rank() == 0:
                # Ensure that only local rank 0 download the dataset
                idist.barrier()
        self.inception = torch.jit.load(inception_path, map_location=self.device)
        self.inception.to(self.device)
        self.inception.eval()

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        data = data.to(self.device)
        data = (data * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return self.inception(data, return_features=True)


class InceptionExtractor:
    def __init__(self) -> None:
        try:
            from torchvision import models
        except ImportError:
            raise RuntimeError("This module requires torchvision to be installed.")
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()

    @torch.no_grad()
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.dim() != 4:
            raise ValueError(f"Inputs should be a tensor of dim 4, got {data.dim()}")
        if data.shape[1] != 3:
            raise ValueError(f"Inputs should be a tensor with 3 channels, got {data.shape}")
        return self.model(data)


class FID(Metric):
    def __init__(
            self,
            precomputed_pkl: Optional[Union[str, Path]] = None,
            computed_pkl_save_path: Optional[Union[str, Path]] = None,
            output_transform: Callable = lambda x: x,
            max_num_examples: Optional[int] = None,
            num_features: Optional[int] = None,
            inception_path="./pretrained_models/stylegan2-ada-fid-inception.pt",
            feature_extractor: Optional[Callable] = None,
            device: Union[str, torch.device] = torch.device("cpu"),
    ) -> None:

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise RuntimeError("This module requires scipy to be installed.")

        # default is inception
        if num_features is None and feature_extractor is None:
            num_features = 2048
            feature_extractor = StyleGAN2InceptionExtractor(inception_path)
        elif num_features is None:
            raise ValueError("Argument num_features should be defined")
        elif feature_extractor is None:
            self._feature_extractor = lambda x: x
            feature_extractor = self._feature_extractor

        if num_features <= 0:
            raise ValueError(f"Argument num_features must be greater to zero, got: {num_features}")
        self._num_features = num_features
        self._feature_extractor = feature_extractor
        self._eps = 1e-6

        if precomputed_pkl is None and computed_pkl_save_path is None:
            raise ValueError(f"must set one of precomputed_pkl, computed_pkl_save_path")

        self.computed_pkl_save_path = computed_pkl_save_path
        if precomputed_pkl is None:
            self._mean, self._cov = None, None
        else:
            self._mean, self._cov = self._load_precomputed_pkl(precomputed_pkl)

        self.max_num_examples = max_num_examples
        super(FID, self).__init__(output_transform=output_transform, device=device)

    @staticmethod
    def _load_precomputed_pkl(precomputed_pkl):
        assert Path(precomputed_pkl).exists(), f"{precomputed_pkl} do not exist"
        with open(precomputed_pkl, "rb") as f:
            reference = pickle.load(f)
            mean = reference['mean']
            cov = reference['cov']
        return mean, cov

    def _online_update(self, features: torch.Tensor) -> None:
        features = features.to(torch.float64)

        if self.raw_mean is None or self.raw_cov is None:
            self.raw_mean = features.sum(dim=0)
            self.raw_cov = features.T @ features
            return

        self.raw_mean += features.sum(dim=0)
        self.raw_cov += features.T @ features
        return

    @staticmethod
    def _check_feature_input(feature: torch.Tensor) -> None:
        if feature.dim() != 2:
            raise ValueError(f"Features must be a tensor of dim 2, got: {feature.dim()}")
        if feature.shape[0] == 0:
            raise ValueError(f"Batch size should be greater than one, got: {feature.shape[0]}")
        if feature.shape[1] == 0:
            raise ValueError(f"Feature size should be greater than one, got: {feature.shape[1]}")

    @reinit__is_reduced
    def reset(self) -> None:
        self.raw_mean, self.raw_cov = None, None
        self._num_examples = 0
        self._last_features = None
        super(FID, self).reset()

    def _update(self, features):
        # Updates the mean and covariance for features
        self._online_update(features)
        self._num_examples += features.shape[0]

    @property
    def is_full(self):
        return self._last_features is not None

    @reinit__is_reduced
    def update(self, output: torch.Tensor) -> None:
        if self._last_features is not None:
            return

        # Extract the features from the outputs
        features = self._feature_extractor(output.detach()).to(self._device)
        cur_num_examples = features.shape[0]
        # Check the feature shapes
        self._check_feature_input(features)
        if self.max_num_examples is None or self.max_num_examples > (
                self._num_examples + cur_num_examples) * idist.get_world_size():
            self._update(features)
        else:
            self._last_features = idist.all_gather(features)

    @sync_all_reduce("_num_examples", "raw_mean", "raw_cov")
    def compute(self):
        if self._last_features is not None:
            _num = self.max_num_examples - self._num_examples
            if _num < 0:
                raise RuntimeError(f"max items: {self.max_num_examples} but now we have: {self._num_examples}")
            self._update(self._last_features[:_num])

        if self.max_num_examples is not None:
            assert self._num_examples == self.max_num_examples, \
                f"num_examples: {self._num_examples} != {self.max_num_examples}"

        cur_mean = (self.raw_mean / self._num_examples).cpu().numpy()
        cur_cov = (self.raw_cov / self._num_examples).cpu().numpy()
        cur_cov = cur_cov - np.outer(cur_mean, cur_mean)

        if self.computed_pkl_save_path is not None and idist.get_rank() == 0:
            with open(self.computed_pkl_save_path, "wb") as f:
                embeds = dict(mean=cur_mean, cov=cur_cov)
                pickle.dump(embeds, f)

        if self._mean is not None and self._cov is not None:
            fid = fid_score_(cur_mean, cur_cov, self._mean, self._cov, self._eps)
            if fid == float("inf"):
                warnings.warn("The product of covariance of train and test features is out of bounds.")
            return fid
        else:
            return float("inf")
