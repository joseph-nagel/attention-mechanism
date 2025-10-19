# Attention mechanism

A short introduction to attention in deep neural networks is provided.
The important components of the classical transformer architecture are discussed.
Following this, a vision transformer (ViT) is implemented and briefly tested.

<img src="assets/attention.svg" alt="The scaled dot-product (cross) attention mechanism is visualized" title="Scaled dot-product (cross) attention" height="250">


## Notebooks

- [Introduction](notebooks/intro.ipynb)
- [(Fashion) MNIST example](notebooks/vit.ipynb)


## Installation

```bash
pip install -e .
```


## ViT training

```bash
python scripts/main.py fit --config config/vit_mnist.yaml
```

```bash
python scripts/main.py fit --config config/vit_fmnist.yaml
```
