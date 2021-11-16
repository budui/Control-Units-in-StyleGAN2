import time
from collections import defaultdict
from pathlib import Path

import fire
import ignite.distributed as idist
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn import svm
from tqdm import tqdm

import raydl
import training.distributed
from editing.config import CLASSIFIER_CKP
from editing.modification import Manipulator
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator
from third_party.attributes_classifier import AttrClassifier


def load_generator(checkpoint, truncation=0.7):
    g = ImprovedStyleGAN2Generator.load(checkpoint, device=idist.device(), default_truncation=truncation)
    g.manipulation_mode()
    g.eval()
    g = training.distributed.auto_model(g)
    return g


class Labor:
    def __init__(self, name):
        self.dim = 0
        self.name = name

    def label(self, images) -> torch.Tensor:
        raise NotImplementedError


class A40Labor(Labor):
    def __init__(self, ckp=CLASSIFIER_CKP):
        super().__init__("a40")
        self.classifier = AttrClassifier(ckp)
        self.dim = 40

    def label(self, images) -> torch.Tensor:
        images = F.interpolate(images, (224, 224))
        return self.classifier.infer(images, logit=True)


class AgeLabor(Labor):
    def __init__(self, ckp=""):
        super(AgeLabor, self).__init__("age")
        jit_model = torch.jit.load(ckp)
        jit_model.to(idist.device())
        jit_model.eval()

        self.model = jit_model
        self.dim = 1

    def label(self, images) -> torch.Tensor:
        images = F.interpolate(images, (128, 128))
        age = torch.zeros([len(images), 1], device=idist.device())
        for i in range(len(images)):
            image = images[i:i + 1, (2, 1, 0)]
            age[i] = self.model(image)[0][0] * 100
        return age


def make_latent_image_generator(checkpoint, truncation=0.7):
    g = load_generator(checkpoint, truncation)

    def worker(batch):
        z = torch.randn(batch, g.style_dim, device=idist.device())
        w = g.z_to_w(z, truncation=truncation)
        image = g.w_to_image(w)
        return w, image

    return worker


def label(output_path, checkpoint, labor_names=("a40",), num_samples=500000, batch_size=32,
          temp_chunk_size=10000):
    assert set(labor_names).issubset(["age", "a40"])
    latent_image_generator = make_latent_image_generator(checkpoint)

    labors = []
    if "a40" in labor_names:
        labors.append(A40Labor())
    if "age" in labor_names:
        labors.append(AgeLabor())
    assert len(labors) > 0
    logger.info(f"labors: {[f'{l.name}:{l.dim}' for l in labors]}")

    record_files = []
    output_path = Path(output_path)
    temp_folder = output_path.parent
    if not temp_folder.exists():
        temp_folder.mkdir()
    records = []
    total = 0
    with torch.no_grad():
        pbar = tqdm(raydl.total_chunk(num_samples, batch_size), total=num_samples // batch_size + 1, ncols=80)
        for batch in pbar:
            latent, images = latent_image_generator(batch)
            scores = []
            for labor in labors:
                score = labor.label(images)
                raydl.assert_shape(score, [images.size(0), labor.dim])
                scores.append(score)

            scores = torch.cat(scores, dim=1)

            records.append((latent, scores))
            total += batch

            if len(records) * batch > temp_chunk_size or total >= num_samples:
                save_path = temp_folder / f"{output_path.name}_{total}.npz"
                np.savez(
                    save_path,
                    w=torch.cat([r[0] for r in records]).cpu().numpy(),
                    score=torch.cat([r[-1] for r in records]).cpu().numpy(),
                )
                records = []
                record_files.append(save_path)

    records = defaultdict(list)
    for sp in record_files:
        rds = np.load(str(sp))
        for k, v in rds.items():
            records[k].append(v)
        sp.unlink()
    data = {}
    for k in records:
        data[k] = np.concatenate(records[k])
    np.savez(output_path, **data)


def train_boundary(latent_codes, scores, chosen_num_or_ratio=0.02, split_ratio=0.7, invalid_value=None):
    """Trains boundary in latent space with offline predicted attribute scores.
    Given a collection of latent codes and the attribute scores predicted from the
    corresponding images, this function will train a linear SVM by treating it as
    a bi-classification problem. Basically, the samples with highest attribute
    scores are treated as positive samples, while those with lowest scores as
    negative. For now, the latent code can ONLY be with 1 dimension.
    NOTE: The returned boundary is with shape (1, latent_space_dim), and also
    normalized with unit norm.
    Args:
      latent_codes: Input latent codes as training data. [num_samples, latent_dim]
      scores: Input attribute scores used to generate training labels. [num_samples, 1]
      chosen_num_or_ratio: How many samples will be chosen as positive (negative)
        samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
        latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
        0.5 * latent_codes_num)` will be used. (default: 0.02)
      split_ratio: Ratio to split training and validation sets. (default: 0.7)
      invalid_value: This field is used to filter out data. (default: None)
    Returns:
      A decision boundary with type `numpy.ndarray`.
    Raises:
      ValueError: If the input `latent_codes` or `scores` are with invalid format.
    """

    if not isinstance(latent_codes, np.ndarray) or not len(latent_codes.shape) == 2:
        raise ValueError(
            f"Input `latent_codes` should be with type"
            f"`numpy.ndarray`, and shape [num_samples, latent_space_dim]!"
        )
    num_samples = latent_codes.shape[0]
    latent_space_dim = latent_codes.shape[1]
    if (
            not isinstance(scores, np.ndarray)
            or not len(scores.shape) == 2
            or not scores.shape[0] == num_samples
            or not scores.shape[1] == 1
    ):
        raise ValueError(
            f"Input `scores` should be with type `numpy.ndarray`, and "
            f"shape [num_samples, 1], where `num_samples` should be "
            f"exactly same as that of input `latent_codes`!"
        )
    if chosen_num_or_ratio <= 0:
        raise ValueError(f"Input `chosen_num_or_ratio` should be positive, " f"but {chosen_num_or_ratio} received!")

    logger.info(f"Filtering training data.")
    if invalid_value is not None:
        latent_codes = latent_codes[scores[:, 0] != invalid_value]
        scores = scores[scores[:, 0] != invalid_value]

    logger.info(f"Sorting scores to get positive and negative samples.")
    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]
    num_samples = latent_codes.shape[0]
    if 0 < chosen_num_or_ratio <= 1:
        chosen_num = int(num_samples * chosen_num_or_ratio)
    else:
        chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, num_samples // 2)

    logger.info(f"Spliting training and validation sets:")
    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num
    # Positive samples.
    positive_idx = np.arange(chosen_num)
    np.random.shuffle(positive_idx)
    positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
    positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
    logger.debug(
        "positive label min:{} max:{} avg:{}".format(
            str(np.min(scores[:chosen_num])), str(np.max(scores[:chosen_num])), str(np.mean(scores[:chosen_num]))
        )
    )
    if np.min(scores[:chosen_num]) < 0.9:
        logger.error("positive samples not enough!")
        return None, None, None

    # Negative samples.
    negative_idx = np.arange(chosen_num)
    np.random.shuffle(negative_idx)
    negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
    negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
    logger.debug(
        "negative label min:{} max:{} avg:{}".format(
            str(np.min(scores[-chosen_num:])), str(np.max(scores[-chosen_num:])), str(np.mean(scores[-chosen_num:]))
        )
    )
    if np.max(scores[-chosen_num:]) > 0.1:
        logger.error("negative samples not enough!")
        return None, None, None
    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate([np.ones(train_num, dtype=np.int64), np.zeros(train_num, dtype=np.int64)], axis=0)
    logger.info(f"  Training: {train_num} positive, {train_num} negative.")
    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate([np.ones(val_num, dtype=np.int64), np.zeros(val_num, dtype=np.int64)], axis=0)
    logger.info(f"  Validation: {val_num} positive, {val_num} negative.")
    # Remaining set.
    remaining_num = num_samples - chosen_num * 2
    remaining_data = latent_codes[chosen_num:-chosen_num]
    remaining_scores = scores[chosen_num:-chosen_num]
    decision_value = (scores[0] + scores[-1]) / 2
    remaining_label = np.ones(remaining_num, dtype=np.int64)
    remaining_label[remaining_scores.ravel() < decision_value] = 0
    remaining_positive_num = np.sum(remaining_label == 1)
    remaining_negative_num = np.sum(remaining_label == 0)
    logger.info(f"  Remaining: {remaining_positive_num} positive, " f"{remaining_negative_num} negative.")

    logger.info(f"Training boundary.")
    start_time = time.time()
    clf = svm.SVC(kernel="linear", verbose=True)
    classifier = clf.fit(train_data, train_label)
    logger.info(f"Finish training. after {time.time() - start_time}s")

    boundary = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    boundary = boundary / np.linalg.norm(boundary)
    validation_acc = None
    remaining_acc = None

    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        logger.info(
            f"Accuracy for validation set: " f"{correct_num} / {val_num * 2} = " f"{correct_num / (val_num * 2):.6f}"
        )
        validation_acc = correct_num / (val_num * 2)

    if remaining_num:
        remaining_prediction = classifier.predict(remaining_data)
        correct_num = np.sum(remaining_label == remaining_prediction)
        logger.info(
            f"Accuracy for remaining set: " f"{correct_num} / {remaining_num} = " f"{correct_num / remaining_num:.6f}"
        )
        remaining_acc = correct_num / remaining_num

    return dict(boundary=boundary, validation_acc=validation_acc, remaining_acc=remaining_acc)


def train(record_path, output_path, checkpoint, score_id=0, chosen_num_or_ratio=0.02):
    records = np.load(record_path)
    latent_code = records["w"]
    score = records["score"]

    logger.info(f"latent_code shape: {latent_code.shape}")
    logger.info(f"score shape: {score.shape}")

    result_dict = train_boundary(latent_code, score[:, score_id: score_id + 1], chosen_num_or_ratio, split_ratio=0.7)

    if result_dict["boundary"] is None:
        logger.error("boundary is None")
    boundary = result_dict["boundary"]
    boundary = torch.from_numpy(boundary).to(torch.float)
    boundary /= boundary.norm(2)

    m = Manipulator()
    with torch.no_grad():
        g = load_generator(checkpoint)
        delta_styles = g.w_to_styles(boundary.to(idist.device()))
        zero_styles = g.w_to_styles(torch.zeros_like(boundary.to(idist.device())))

        for layer, zero, delta in zip(range(len(zero_styles)), zero_styles, delta_styles):
            m.add_movement(layer, delta - zero)
        m.save(output_path)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    fire.Fire(dict(train=train, label=label))
