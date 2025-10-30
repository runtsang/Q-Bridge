"""Classical classifier mirroring the quantum helper interface with optional quantum feature map.

The module defines a single class `QuantumClassifierModel` that can be instantiated
with a specified number of features and depth.  It constructs a feed‑forward
network of `nn.Linear` layers followed by `nn.ReLU`.  The depth controls the
number of hidden layers and the total number of trainable parameters.
An optional `feature_map` callable can be supplied; if provided, the input
features are first transformed by this callable before being fed into the
network.  The class also offers a convenience method to build a Qiskit
`EstimatorQNN` that mirrors the network's structure, enabling a direct
comparison between the classical and quantum implementations.

The design follows the “combination” scaling paradigm: the classical model
scales by increasing depth or feature dimension, while the quantum model
scales by increasing circuit depth or qubit count.  The shared API makes
experimentation across both worlds straightforward.

"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  Classical network factory
# --------------------------------------------------------------------------- #
def build_classifier_network(
    num_features: int,
    depth: int,
    feature_map: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Tuple[nn.Module, List[int]]:
    """
    Construct a feed‑forward classifier and return the network together
    with a list of layer sizes for introspection.

    Parameters
    ----------
    num_features: int
        Number of input features.
    depth: int
        Number of hidden layers.  Each hidden layer has the same width
        as the input dimension.
    feature_map: Callable | None
        Optional callable that transforms the raw input tensor before
        it enters the network.

    Returns
    -------
    network: nn.Module
        The assembled sequential network.
    layer_sizes: List[int]
        Number of parameters in each linear layer (weights + bias).
    """
    layers = []
    in_dim = num_features
    layer_sizes: List[int] = []

    if feature_map is not None:
        # Wrap the feature map in a lambda that can be used as a module
        class FeatureMapModule(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return feature_map(x)

        layers.append(FeatureMapModule())

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        layer_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    layer_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    return network, layer_sizes


# --------------------------------------------------------------------------- #
#  Class exposing training utilities
# --------------------------------------------------------------------------- #
class QuantumClassifierModel(nn.Module):
    """
    A lightweight wrapper around a classical feed‑forward classifier.

    The class is intentionally minimal; it delegates the heavy lifting to
    :func:`build_classifier_network`.  It also provides a static method
    that returns a Qiskit EstimatorQNN with an equivalent architecture,
    enabling a side‑by‑side benchmark.
    """

    def __init__(
        self,
        num_features: int,
        depth: int,
        feature_map: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.network, self.layer_sizes = build_classifier_network(
            num_features, depth, feature_map
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @staticmethod
    def to_qiskit_estimator(
        num_features: int,
        depth: int,
        backend="qasm_simulator",
    ):
        """
        Build a Qiskit EstimatorQNN that mirrors the classical architecture.
        The quantum circuit consists of a data‑re‑uploading ansatz with the
        same number of qubits as input features and depth equal to ``depth``.
        Each qubit is measured in the Z basis.

        Parameters
        ----------
        num_features: int
            Number of qubits / input features.
        depth: int
            Number of variational layers.
        backend: str
            Backend name for the Qiskit estimator.

        Returns
        -------
        estimator_qnn: qiskit_machine_learning.neural_networks.EstimatorQNN
            A compiled EstimatorQNN ready for training.
        """
        from qiskit import QuantumCircuit, Aer
        from qiskit.circuit import ParameterVector
        from qiskit.quantum_info import SparsePauliOp
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit.primitives import Estimator

        # Encoding and variational parameters
        encoding = ParameterVector("x", num_features)
        weights = ParameterVector("theta", num_features * depth)

        qc = QuantumCircuit(num_features)

        # Data re‑uploading: encode each feature once per depth
        for param, qubit in zip(encoding, range(num_features)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(num_features):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(num_features - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_features - i - 1))
            for i in range(num_features)
        ]

        estimator = Estimator(backend=Aer.get_backend(backend))
        estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observables,
            input_params=encoding,
            weight_params=weights,
            estimator=estimator,
        )
        return estimator_qnn

__all__ = ["QuantumClassifierModel", "build_classifier_network"]
