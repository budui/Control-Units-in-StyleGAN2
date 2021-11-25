# Control-Units-in-StyleGAN2

[Project](https://wrong.wang/x/Control-Units-in-StyleGAN2/) | [Paper](https://dl.acm.org/doi/10.1145/3474085.3475274)

The official PyTorch implementation for MM'21 paper 'Attribute-specific Control Units in StyleGAN for Fine-grained Image Manipulation'


We are planning to publish a detailed tutorial on how to use it in a few days.

Take a look first:

```bash
# Download pretrained StyleGAN2 model: https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing
# And put it in ./pretrained
# Run this command and then see the editing result in ./tmp
python3 manipulation.py test ./pretrained/modifications/before_alter_Black_Hair_12.mdfc --max_factor 20
```

# Pretrained Models

Please download the pre-trained models from the following link

[Google Drive](https://drive.google.com/drive/folders/1g-ukOZ_KZXHSroLXq87jTx7iTlIinhzf?usp=sharing)



# Citation

If you use this code for your research, please cite our paper [Attribute-specific Control Units in StyleGAN for Fine-grained Image Manipulation
](https://dl.acm.org/doi/10.1145/3474085.3475274)

```text
@inproceedings{10.1145/3474085.3475274,
author = {Wang, Rui and Chen, Jian and Yu, Gang and Sun, Li and Yu, Changqian and Gao, Changxin and Sang, Nong},
title = {Attribute-Specific Control Units in StyleGAN for Fine-Grained Image Manipulation},
year = {2021},
isbn = {9781450386517},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3474085.3475274},
doi = {10.1145/3474085.3475274},
booktitle = {Proceedings of the 29th ACM International Conference on Multimedia},
pages = {926–934},
numpages = {9},
keywords = {generative adversarial networks(GANs), control unit, image manipulation},
location = {Virtual Event, China},
series = {MM '21}
}

```
