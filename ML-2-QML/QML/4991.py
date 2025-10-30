"""Quantum implementation of ConvGen260. Mirrors the classical interface but uses a
variational circuit with random layers and a threshold mapping that emulates the
classical sigmoid activation.  The class also exposes helper functions for
building a classifier circuit and a simple estimator QNN, enabling a direct
comparison with the classical counterpart."""
