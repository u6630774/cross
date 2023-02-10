# pytorch-fcn

[![PyPI Version](https://img.shields.io/pypi/v/torchfcn.svg)](https://pypi.python.org/pypi/torchfcn)
[![Python Versions](https://img.shields.io/pypi/pyversions/torchfcn.svg)](https://pypi.org/project/torchfcn)
[![GitHub Actions](https://github.com/wkentaro/pytorch-fcn/workflows/CI/badge.svg)](https://github.com/wkentaro/pytorch-fcn/actions)

PyTorch implementation of [Fully Convolutional Networks](https://github.com/shelhamer/fcn.berkeleyvision.org).


## Requirements

- [pytorch](https://github.com/pytorch/pytorch) >= 0.2.0
- [torchvision](https://github.com/pytorch/vision) >= 0.1.8
- [fcn](https://github.com/wkentaro/fcn) >= 6.1.5
- [Pillow](https://github.com/python-pillow/Pillow)
- [scipy](https://github.com/scipy/scipy)
- [tqdm](https://github.com/tqdm/tqdm)


## Installation

```bash
git clone https://github.com/wkentaro/pytorch-fcn.git
cd pytorch-fcn
pip install .

# or

pip install torchfcn
```

## Cite This Project

If you use this project in your research or wish to refer to the baseline results published in the README, please use the following BibTeX entry.

```bash
@misc{pytorch-fcn2017,
  author =       {Ketaro Wada},
  title =        {{pytorch-fcn: PyTorch Implementation of Fully Convolutional Networks}},
  howpublished = {\url{https://github.com/wkentaro/pytorch-fcn}},
  year =         {2017}
}
```
