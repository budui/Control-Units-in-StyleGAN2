import os
import sys
from pathlib import Path

import dlib
import fire
import numpy as np
import scipy
import scipy.ndimage
from PIL import Image
from ignite.handlers import Timer
from loguru import logger
from tqdm import tqdm

import raydl


def orthogonal_vector(v):
    return np.flipud(v) * [-1, 1]


def oriented_crop_rectangle(left_eye, right_eye, mouth, scale_ratio, quad_ratio):
    eye_middle = (left_eye + right_eye) / 2
    l_eye_to_r_eye = right_eye - left_eye
    eye_to_mouth = mouth - eye_middle

    x = l_eye_to_r_eye - orthogonal_vector(eye_to_mouth)
    x = x / np.hypot(*x)
    x *= max(np.hypot(*l_eye_to_r_eye) * 2.0 * scale_ratio, np.hypot(*eye_to_mouth) * 1.8 * scale_ratio)

    y = orthogonal_vector(x)
    c = eye_middle + eye_to_mouth * 0.1

    quadrangle = np.stack([
        c - quad_ratio[0] * x - quad_ratio[1] * y,
        c - quad_ratio[0] * x + quad_ratio[3] * y,
        c + quad_ratio[2] * x + quad_ratio[3] * y,
        c + quad_ratio[2] * x - quad_ratio[1] * y
    ])
    quadrangle_size = np.array((
        np.hypot(*x) * (quad_ratio[0] + quad_ratio[2]),
        np.hypot(*y) * (quad_ratio[1] + quad_ratio[3])
    ))
    return quadrangle, quadrangle_size


def shrink_image(img, quadrangle, quadrangle_size, output_width, output_height):
    shrink = min(
        int(np.floor(quadrangle_size[0] / output_width * 0.5)),
        int(np.floor(quadrangle_size[1] / output_height * 0.5))
    )
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quadrangle /= shrink
        quadrangle_size /= shrink
    return img, quadrangle, quadrangle_size


def crop_image(img, quadrangle, quadrangle_size):
    border_w = max(int(np.rint(quadrangle_size[0] * 0.1)), 3)
    border_h = max(int(np.rint(quadrangle_size[1] * 0.1)), 3)
    crop = (
        int(np.floor(min(quadrangle[:, 0]))),
        int(np.floor(min(quadrangle[:, 1]))),
        int(np.ceil(max(quadrangle[:, 0]))),
        int(np.ceil(max(quadrangle[:, 1])))
    )
    crop = (
        max(crop[0] - border_w, 0),
        max(crop[1] - border_h, 0),
        min(crop[2] + border_w, img.size[0]),
        min(crop[3] + border_h, img.size[1])
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quadrangle -= crop[0:2]
    return img, quadrangle


def pad_image(img, quadrangle, quadrangle_size, padding_mode="reflect", blur_padding=True, save_padding_ratio=False):
    border_w = max(int(np.rint(quadrangle_size[0] * 0.1)), 3)
    border_h = max(int(np.rint(quadrangle_size[1] * 0.1)), 3)
    pad = (
        int(np.floor(min(quadrangle[:, 0]))),
        int(np.floor(min(quadrangle[:, 1]))),
        int(np.ceil(max(quadrangle[:, 0]))),
        int(np.ceil(max(quadrangle[:, 1])))
    )
    pad = (
        max(-pad[0] + border_w, 0),
        max(-pad[1] + border_h, 0),
        max(pad[2] - img.size[0] + border_w, 0),
        max(pad[3] - img.size[1] + border_h, 0)
    )
    pad_img = None
    if save_padding_ratio:
        pad_img = np.ones_like(np.float32(img))
    if max(pad) > border_w - 4 or max(pad) > border_h - 4:
        pad = np.array((
            max(pad[0], int(np.rint(quadrangle_size[0] * 0.3))),  # left
            max(pad[1], int(np.rint(quadrangle_size[1] * 0.3))),  # top
            max(pad[2], int(np.rint(quadrangle_size[0] * 0.3))),  # right
            max(pad[3], int(np.rint(quadrangle_size[1] * 0.3)))  # down
        ))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), padding_mode)
        if save_padding_ratio:
            pad_img = np.pad(pad_img, ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), "constant")
        if blur_padding:
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(
                1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3])
            )
            blur = quadrangle_size * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [*blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)

        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quadrangle += pad[:2]

    if save_padding_ratio:
        pad_img = Image.fromarray(np.uint8(np.rint(pad_img)))
    return img, quadrangle, pad_img


def align_face_with_eyes_mouth(
        img: Image,
        left_eye, right_eye, mouth,
        scale_ratio=1.0,
        quadrangle_ratio=1.0,
        enable_padding=True,
        output_width=1024,
        padding_mode="reflect",
        blur_padding=True,
        save_padding_ratio=True,
):
    if isinstance(quadrangle_ratio, (float, int)):
        quadrangle_ratio = [quadrangle_ratio, ]
    assert len(quadrangle_ratio) in [1, 2, 4]
    quadrangle_ratio *= 4 // len(quadrangle_ratio)
    quadrangle_ratio = np.array(quadrangle_ratio)
    output_height = int(output_width * (
            (quadrangle_ratio[1] + quadrangle_ratio[3]) / (quadrangle_ratio[0] + quadrangle_ratio[2])
    ) + 0.5)
    logger.debug(f"output_height: {output_height} output_width: {output_width}")
    logger.debug(f"quadrangle_ratio(lw, uh, rw, dh): {quadrangle_ratio}")

    quadrangle, quadrangle_size = oriented_crop_rectangle(left_eye, right_eye, mouth, scale_ratio, quadrangle_ratio)
    logger.debug(f"quadrangle_size:{quadrangle_size}, quadrangle: {quadrangle}")

    img, quadrangle, quadrangle_size = shrink_image(img, quadrangle, quadrangle_size, output_width, output_height)
    img, quadrangle = crop_image(img, quadrangle, quadrangle_size)
    pad_img = None
    if enable_padding:
        img, quadrangle, pad_img = pad_image(img, quadrangle, quadrangle_size, padding_mode,
                                             blur_padding=blur_padding, save_padding_ratio=save_padding_ratio)

    img = img.transform(
        (output_width, output_height),
        Image.QUAD,
        (quadrangle + 0.5).flatten(),
        Image.BILINEAR
    )
    padding_ratio = 0
    if save_padding_ratio:
        pad_img = np.array(pad_img.transform(
            (output_width, output_height),
            Image.QUAD,
            (quadrangle + 0.5).flatten(),
            Image.BILINEAR
        ))
        padding_ratio = 1 - pad_img.sum() / pad_img.size

    return img, padding_ratio


class DlibLandmark:
    def __init__(
            self,
            shape_predictor="./pretrained_models/shape_predictor_68_face_landmarks.dat"
    ):
        self.predictor = dlib.shape_predictor(shape_predictor)
        self.detector = dlib.get_frontal_face_detector()

    def predict(self, image_path):
        img = dlib.load_rgb_image(image_path)
        dets = self.detector(img, 1)
        if len(dets) == 0:
            return None
        shape = None
        for _, d in enumerate(dets):
            shape = self.predictor(img, d)

        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)

        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        lm_mouth_outer = lm[48: 60]  # left-clockwise
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth = (mouth_left + mouth_right) * 0.5
        return dict(left_eye=eye_left, right_eye=eye_right, mouth=mouth)


class Youtu87Landmark:
    def __init__(self, pts_path):
        self.pts_path = Path(pts_path)

    def predict(self, image_file):
        pts_file = self.pts_path / f"{Path(image_file).name}.txt"
        if not pts_file.exists():
            return None
        with open(pts_file, 'r') as fin:
            landmarks = fin.readlines()
            face_landmarks = np.asarray([np.fromstring(n.strip(), dtype=float, sep=',') for n in landmarks[:87]])

        lm = np.array(face_landmarks)
        lm_eye_left = lm[35: 43]  # left-clockwise
        lm_eye_right = lm[45: 53]  # left-clockwise
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        mouth_left = lm[65]
        mouth_right = lm[66]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        return dict(left_eye=eye_left, right_eye=eye_right, mouth=mouth_avg)


def main(
        root,
        save_path,
        pts_path=None,
        record_skip=False,
        scale_ratio=1,
        blur_padding=True,
        padding_mode="reflect",
        quadrangle_ratio=(1.0, 1.0, 1.0, 1.0),
        output_width=1024,
        force=False,
):
    root = Path(root)
    save_path = Path(save_path).resolve().absolute()
    if not save_path.exists():
        save_path.mkdir()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    if pts_path is None:
        dl = DlibLandmark()
    else:
        dl = Youtu87Landmark(pts_path)
    logger.info("Landmark predictor over")
    timer = Timer()
    padding_ratio_file = open(save_path / f"padding_ratio.txt", "w", buffering=1)

    images = raydl.images_files(root)
    images = sorted(images, key=os.path.getctime)
    pbar = tqdm(images, ncols=80)

    for p in pbar:
        if (save_path / p.name).exists() and not force:
            continue
        eyes_mouth = dl.predict(str(p))
        if eyes_mouth is None:
            logger.warning(f"can not find face, skip {p}")
            if record_skip:
                with open(save_path / "skip.txt", "a") as out:
                    print(p.as_posix(), file=out)
            continue
        face, padding_ratio = align_face_with_eyes_mouth(
            Image.open(p), **eyes_mouth,
            scale_ratio=scale_ratio,
            blur_padding=blur_padding,
            padding_mode=padding_mode,
            quadrangle_ratio=quadrangle_ratio,
            output_width=output_width
        )
        pbar.write(f"Aligned image {p} {face.size} {timer.value():.2f}s, padding_ratio:{padding_ratio:.4f}")
        print(f"{str(save_path / p.name)},{padding_ratio:.6f}", file=padding_ratio_file)
        face.save(save_path / p.name)
        timer.reset()
    padding_ratio_file.close()


if __name__ == '__main__':
    fire.Fire(main)
