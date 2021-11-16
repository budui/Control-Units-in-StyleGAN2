import fire
import ignite.distributed as idist
import torch
from loguru import logger
from tqdm import tqdm

from editing.config import FACE_PARSER_CKP
from models.StyleGAN2_wrapper import ImprovedStyleGAN2Generator
from third_party.BiSetNet import FaceParser


def main(checkpoint, save_path, num_samples=10000, batch_size=8, truncation=0.7):
    device = idist.device()
    G = ImprovedStyleGAN2Generator.load(checkpoint, device=device, default_truncation=truncation)
    G.manipulation_mode()

    logger.info("load generator over")

    face_parser = FaceParser(model_path=FACE_PARSER_CKP, device=device)

    logger.info("load face_parser over")
    num_batch = (num_samples + batch_size - 1) // batch_size
    batch_id = num_batch

    style_grads = None
    style_grad_num = None

    pbar = tqdm(total=num_batch, ncols=0)

    while batch_id > 0:
        z = torch.randn(batch_size, G.style_dim, device=device)
        w = G.z_to_w(z)
        styles = G.w_to_styles(w)

        styles = [s.detach().requires_grad_(True) for s in styles]
        images = G.styles_to_image(styles)

        with torch.no_grad():
            parsing = face_parser.batch_run(images, pre_normalize=True, image_repr=False, compact_mask=True)
            if parsing is None:
                continue

        if style_grads is None:
            style_grads = [[torch.zeros(s.size(-1), device=device) for _ in range(parsing.size(1))] for s in styles]
            style_grad_num = [[0 for _ in range(parsing.size(1))] for _ in styles]

        for mask_id in range(parsing.size(1)):
            G.zero_grad()
            for s in styles:
                s.grad = None
            grad_map = parsing[:, [mask_id, ]].repeat(1, 3, 1, 1).float()
            grad_map /= grad_map.abs().sum(dim=[1, 2, 3], keepdim=True).clip_(1e-5)

            # some mask result may not contains any content, e.g. full of 0.
            num_valid = (grad_map.sum(dim=[-1, -2, -3]) > 0).sum()
            images.backward(grad_map, retain_graph=True)

            for i, s in enumerate(styles):
                style_grads[i][mask_id] += s.grad.abs().sum(dim=[0])
                style_grad_num[i][mask_id] += num_valid

        batch_id -= 1
        pbar.update(1)
    pbar.close()

    channel_correction = []
    print(','.join(map(str, [float(c) / (num_batch * batch_size) for c in style_grad_num[0]])))
    for layer in range(len(style_grads)):
        channel_correction.append(torch.stack([c.div(n) for c, n in zip(style_grads[layer], style_grad_num[layer])]))
    torch.save(channel_correction, save_path)


if __name__ == '__main__':
    fire.Fire(main)
