# Attention mechanism

A short introduction to attention in deep neural networks is provided.
The important components of the classical transformer architecture are discussed.
Following this, a vision transformer (ViT) is implemented and briefly tested.

<img src="assets/sinusoidal.svg" alt="Sinusoidal encoding of spatial positions or times" title="Sinusoidal encoding" height="300">


## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [(Fashion) MNIST example](notebooks/vit.ipynb)


## Installation

```
pip install -e .
```


## ViT training

```
python scripts/main.py fit --config config/vit_mnist.yaml
```

```
python scripts/main.py fit --config config/vit_fmnist.yaml
```

