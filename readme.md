## Hierarchically-nested Adversarial Network (Pytorch implementation)
###### We call our method **HDGAN**, referring to HighDefinition results and the idea of Hierarchically-nested Discriminators

> [Zizhao Zhang*, Yuanpu Xie*, Lin Yang, "Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network", CVPR (2018)](https://arxiv.org/abs/1802.09178) * indicates contribution

![Discriminator Diagram](/Figures/arch.jpg)


## Dependencies
- python 3
- pytorch
- Anaconda 3.6

## Data
Download our preprocessed data from [birds](https://www.dropbox.com/sh/v0vcgwue2nkwgrf/AACxoRYTAAacmPVfEvY-eDzia?dl=0), and [flowers](https://www.dropbox.com/sh/g8rmz41xblaszb1/AABPNtIcLu1fKNoBsJTHJTIKa?dl=0), save them to Data/birds and Data/flowers, respectively.

## Training
- For bird:   goto train/train_gan:  `bash train_birds.sh`
- For flower: goto train/train_gan:  `bash train_flower.sh`

#### Monitor your training in two ways
- Launch Visdom (see [here](https://github.com/facebookresearch/visdom)): `python -m visdom.server -port 43426` (keep the same with __port_ in plot_utils.py). Then access http://localhost:8097 from the browser.
- Check fixed sample results per epoch in the checkpoint folder.

## Testing
- To generate images, goto test/test_gan:   `bash test_birds.sh` for birds, and  `bash test_flowers.sh` for flower. 

## Evaluation
Evaluation uses the sampled results obtained in Testing and saved in ./Results.
- Go to /Evaluation
- Compute the inception score: `sh compute_inception_score.sh`
- Compute the MS-SSIM score: `sh compute_ms_ssim.sh`
- Evaluate the VS-smilarity score: `sh compute_neudist_score.sh` 


## Pretrained Model
- Download the [pretrained models](https://www.dropbox.com/sh/lpzsvwabkw8d26g/AADFRKpTvbylhl0Q3PH78qzha?dl=0). Save them to Models.
- It contains HDGAN for birds and flowers, visual similarity model for birds and flowers


## Acknowlegement
- StakGAN [tensorflow implementation](https://github.com/hanzhanggit/StackGAN)
- MS-SSIM [Python implementation](https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py)


### Citing StackGAN
If you find HDGAN useful in your research, please consider citing:

```
@inproceedings{hdgan,
Author = {Zizhao Zhang and Yuanpu Xie and Lin Yang},
Title = {Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network},
Year = {2018},
booktitle = {CVPR},
}
```

## License
MIT
