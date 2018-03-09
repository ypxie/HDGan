## Hierarchically-nested Adversarial Network

This is a Pytorch implementation of HD GAN for reproducing main results in the paper:
> [Z.Zhang*, Y. Xie*, L. Yang. "Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network." CVPR (2018)](link)


## Dependencies
python 3

pytorch

Anaconda 3.6


## Data
Download our preprocessed data from [birds](https://www.dropbox.com/sh/v0vcgwue2nkwgrf/AACxoRYTAAacmPVfEvY-eDzia?dl=0), and [flowers](https://www.dropbox.com/sh/g8rmz41xblaszb1/AABPNtIcLu1fKNoBsJTHJTIKa?dl=0), save them to Data/birds and Data/flowers, respectively.

## Training
- For bird: Goto train/train_gan:    `python train_bird.py`
- For flower: Goto train/train_gan:  `python train_flower.py`

## Testing
- To generate images, goto test/test_gan:   `python test_both`
- To evaluate the VS-smilarity, goto test/test_neudist: `python test_neudist.py`

## Pretrained Model
- Download the [pretrained models](https://www.dropbox.com/sh/lpzsvwabkw8d26g/AADFRKpTvbylhl0Q3PH78qzha?dl=0).
Save them to Models.
- It contains `HDGAN for birds and flowers`,  `visual similarity model for birds and flowers`

## Hierarchically-nested Adversarial Network

![Discriminator Diagram](/Figures/disc.png)

## Acknowlegement
StakGAN [tensorflow implementation](https://github.com/hanzhanggit/StackGAN)

### Citing StackGAN
If you find HDGAN useful in your research, please consider citing:

```
@inproceedings{hdgan,
Author = {ZiZhao Zhang and Yuanpu Xie and Lin Yang},
Title = {Photographic Text-to-Image Synthesis with a Hierarchically-nested Adversarial Network},
Year = {2018},
booktitle = {CVPR},
}
```

## License
MIT
