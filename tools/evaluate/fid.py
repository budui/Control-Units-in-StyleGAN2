import sys
from pathlib import Path

import fire
import ignite.distributed as idist
import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Timer
from loguru import logger
from torch.utils.data import Dataset
from torchvision import transforms

import raydl
import training.distributed
from data.dataset import ImageDataset
from raydl.metrics import FID, LifeWise


class FlipDataset(Dataset):
    def __init__(self, dataset):
        self.original = dataset

    def __repr__(self):
        return "Flip" + repr(self.original)

    def __len__(self):
        return len(self.original) * 2

    def __getitem__(self, index):
        if index < len(self.original):
            return self.original[index]
        image = self.original[index - len(self.original)]
        return torch.flip(image, [-1])


def get_dataset(path=None, resolution=None):
    if path is None:
        return None

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = ImageDataset(path, transform, resolution, archive_type=None)
    return dataset


def get_dataloader(dataset=None, xflip=True, batch_size=None, num_workers=None):
    if dataset is None:
        def infinite_iterator():
            while True:
                yield

        return infinite_iterator()

    if xflip:
        logger.info("add xflip argument")
        dataset = FlipDataset(dataset)

    logger.info(f"use dataset: \n{dataset},\n have {len(dataset)} images")

    loader = training.distributed.auto_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False, seed=0,
    )
    return loader


def create_fid_evaluator(pkl_path, inception_path, num_images=None, step_fn=None):
    if not Path(pkl_path).exists():
        logger.info(f"{pkl_path} do not exist. Computed result pkl will be saved at there.")
        computed_pkl_save_path = pkl_path
        precomputed_pkl = None
    else:
        precomputed_pkl = pkl_path
        computed_pkl_save_path = None

    evaluator = Engine(step_fn or (lambda engine, batch: batch))

    fider = FID(precomputed_pkl, computed_pkl_save_path=computed_pkl_save_path, inception_path=inception_path,
                max_num_examples=num_images)
    fider.attach(evaluator, "fid", LifeWise())

    if idist.get_rank() == 0:
        pbar = ProgressBar(ncols=80, )
        pbar.attach(evaluator)

    @evaluator.on(Events.ITERATION_COMPLETED)
    def update(engine: Engine):
        if fider.is_full:
            engine.terminate()

    @evaluator.on(Events.COMPLETED)
    def logg(engine: Engine):
        logger.info("over")
        if precomputed_pkl is not None:
            logger.info(f"metric fid: {engine.state.metrics['fid']:.4f}")
        else:
            logger.info(f"save computed pkl at {computed_pkl_save_path}")

    return evaluator


def generator_step_fn(checkpoint, batch_size=32, truncation=1):
    from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator

    g = ImprovedStyleGAN2Generator.load(checkpoint, device=idist.device(), default_truncation=truncation)
    g.manipulation_mode()
    g.eval()
    g = training.distributed.auto_model(g)

    @torch.no_grad()
    def step(engine, _):
        z = torch.randn(batch_size, g.style_dim, device=idist.device())
        images = g(z=z)
        return images

    return step


def running(
        local_rank,
        pkl_path,
        num_images=None,
        path=None,
        resolution=None,
        batch_size=32,
        num_workers=4,
        xflip=True,
        truncation=1,
        inception_path="./pretrained_models/stylegan2-ada-fid-inception.pt",
):
    logger.remove()
    if local_rank == 0:
        logger.add(sys.stderr, level="DEBUG")

    if Path(path).suffix == ".pt":
        logger.info(f"will load generator from checkpoint {path}")
        dataset = get_dataset()
        dataloader = get_dataloader(dataset)
        step_fn = generator_step_fn(path, batch_size, truncation=truncation)
        num_images = num_images or 50_000
    else:
        logger.info(f"will load images from {path}")
        dataset = get_dataset(path, resolution)
        dataloader = get_dataloader(dataset, xflip=xflip, batch_size=batch_size, num_workers=num_workers)
        step_fn = None

    timer = Timer()
    logger.debug(raydl.describe_dict(dict(
        pkl_path=pkl_path,
        inception_path=inception_path,
        num_images=num_images,
    )), head="evaluator dict:")
    evaluator = create_fid_evaluator(pkl_path, inception_path, num_images, step_fn)
    evaluator.run(dataloader)
    logger.info(f"total running time: {timer.value():.2f}s")


def run(
        pkl_path,
        path=None,
        resolution=None,
        num_images=None,
        batch_size=32,
        num_workers=4,
        xflip=True,
        truncation=1,
        inception_path="./pretrained_models/stylegan2-ada-fid-inception.pt",
        backend="nccl",
):
    kwargs = locals()
    logger.info(raydl.describe_dict(kwargs, head="configs:"))
    with idist.Parallel(backend=kwargs.pop("backend")) as parallel:
        parallel.run(running, **kwargs)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(run)
