# Attention mechanism

![Sinusoidal encoding of spatial positions or times](assets/sinusoidal.svg "Sinusoidal encoding")

A short introduction to attention in deep neural networks is provided.
The important components of the classical transformer architecture are discussed.
Following this, a vision transformer (ViT) is implemented and briefly tested.

## Installation

```
pip install -e .
```

## ViT training

```
python scripts/main.py fit --config config/vit_fmnist.yaml
```

## Notebooks

- [Introduction](notebooks/intro.ipynb)

- [Tensor shapes](notebooks/shapes.ipynb)

