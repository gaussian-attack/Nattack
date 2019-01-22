# Deflecting Adversarial Attacks with Pixel Deflection

![deflecting pixels](https://i.imgur.com/BhxmVwx.png)

Code for paper: https://arxiv.org/abs/1801.08926 

Blog with demo: https://iamaaditya.github.io/2018/02/demo-for-pixel-deflection/

Requirements:

1. Keras 2.0+
(only used for classification - Pixel Deflection itself is deep learning platform independent) 

2. Scipy 1.0+

(Older version of scipy wavelet transform does not have BayesShrink)

* Python3 + tensorflow

## [NATTACK] evaluation

Run with:

```bash
python re_li_attack.py --imagenet-path <path>
````
