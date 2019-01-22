# Countering Adversarial Images using Input Transformations

Paper: [Guo et al. 2018](https://arxiv.org/abs/1711.00117)


## Requirements

* Python3 + tensorflow

## [NATTACK] evaluation

Run with:

```bash
python re_l2_attack_clipimage.py --imagenet-path <path>
````
You can define the defense type in re_l2_attack_clipimage.py
Where `<defense>` is one of `bitdepth`, `jpeg`, `crop`, `quilt`, or `tv`.

[robustml]: https://github.com/robust-ml/robustml
