"""Hybrid classical model that emulates the Quantum‑NAT architecture.

The class is a drop‑in replacement for the original QFCModel but replaces
the quantum sub‑modules with lightweight classical approximations:

* A 2‑layer CNN extracts local image features.
* A 2‑pixel convolutional filter mimics the quanvolution layer.
* A 3‑layer feed‑forward network replaces the quantum fully‑connected layer.
* A final linear head produces a 4‑dimensional output.

The model remains fully differentiable and can be trained with standard
PyTorch optimisers."""
