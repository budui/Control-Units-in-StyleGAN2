import math
from collections import defaultdict
from pathlib import Path

import cv2
import fire
import numpy as np
from skimage.metrics import structural_similarity
from torchvision.datasets.folder import is_image_file
from tqdm import tqdm


def Vollath(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    shape = np.shape(img)
    u = np.mean(img)
    out = -shape[0] * shape[1] * (u ** 2)
    for x in range(0, shape[0] - 1):
        for y in range(0, shape[1]):
            out += int(img[x, y]) * int(img[x + 1, y])
    return out


def entropy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    out = 0
    count = np.shape(img)[0] * np.shape(img)[1]
    p = np.bincount(np.array(img).flatten())
    for i in range(0, len(p)):
        if p[i] != 0:
            out -= p[i] * math.log(p[i] / count) / count
    return out


def sobel(img):
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    return dst


def nrss(image):
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    G, Gr = sobel(image), sobel(image_blur)

    (h, w) = G.shape
    G_blk_list = []
    Gr_blk_list = []
    sp = 6
    for i in range(sp):
        for j in range(sp):
            G_blk = G[int((i / sp) * h):int(((i + 1) / sp) * h),
                    int((j / sp) * w):int(((j + 1) / sp) * w)]
            Gr_blk = Gr[int((i / sp) * h):int(((i + 1) / sp) * h),
                     int((j / sp) * w):int(((j + 1) / sp) * w)]
            G_blk_list.append(G_blk)
            Gr_blk_list.append(Gr_blk)
    sum = 0
    for i in range(sp * sp):
        mssim = structural_similarity(G_blk_list[i], Gr_blk_list[i])
        sum = mssim + sum
    return 1 - sum / (sp * sp * 1.0)


def reblur(img):
    image_blur = cv2.GaussianBlur(img, (7, 7), 0)

    img = img.astype(np.float64)
    image_blur = image_blur.astype(np.float64)

    s_Vver = np.clip(np.abs(img[1:, :-1] - img[:-1, :-1]) - np.abs(
        image_blur[1:, :-1] - image_blur[:-1, :-1]), 0, None).sum()
    s_Fver = np.abs(img[1:, :-1] - img[:-1, :-1]).sum()

    s_Fhor = np.abs(img[:-1, 1:] - img[:-1, :-1]).sum()
    s_Vhor = np.clip(np.abs(img[:-1, 1:] - img[:-1, :-1]) - np.abs(
        image_blur[:-1, 1:] - image_blur[:-1, :-1]), 0, None).sum()

    return 1 - max((s_Fver - s_Vver) / s_Fver, (s_Fhor - s_Vhor) / s_Fhor)


def energy(img):
    img = img.astype(np.float64)
    return np.power(img[1:, :-1] - img[:-1, :-1], 2).sum() / img.size + np.power(img[:-1, 1:] - img[:-1, :-1],
                                                                                 2).sum() / img.size


def SMD2(gray):
    gray = gray.astype(np.float64)
    return (np.abs(gray[1:, :-1] - gray[:-1, :-1]) * np.abs(gray[:-1, 1:] - gray[:-1, :-1])).sum() / gray.size


def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像约清晰越大
    '''
    # shape = np.shape(img)
    # out = 0
    # for x in range(0, shape[0]-2):
    #     for y in range(0, shape[1]):
    #         out += (int(img[x+2, y])-int(img[x, y]))**2
    img = img.astype(np.float64)
    return np.power((img[2:] - img[:-2]), 2).sum() / img.size


def loop(image_path: Path):
    gray = cv2.imread(image_path.as_posix(), cv2.IMREAD_GRAYSCALE)
    return dict(
        SMD2=SMD2(gray),
        brenner=brenner(gray),
        nrss=nrss(gray),
        energy=energy(gray),
        reblur=reblur(gray),
    )


def main(root):
    """
    calculate image 'blur-ness' with various metrics
    :param root: image_folder
    :return:
    """
    metrics = defaultdict(list)
    pbar = tqdm(filter(lambda path: is_image_file(path.name), Path(root).glob("*")))
    for img in pbar:
        r = loop(img)
        desc = ", ".join(f"{k}:{v:.2f}" for k, v in r.items())
        pbar.set_description(f"{desc} - {img.name}")
        for k in r:
            metrics[k].append(r[k])

    print("\n______________________________\n")
    for k in metrics:
        print(k, f"{sum(metrics[k]) / len(metrics[k]):.4f}")


if __name__ == "__main__":
    fire.Fire(main)
