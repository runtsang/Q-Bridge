"""Quantum implementation of a hybrid quanvolution estimator.

This module defines :class:`HybridQuanvolutionEstimator`, a quantum neural
network that mimics the classical architecture from ``Quanvolution.py``.
It consists of:
  1. A *quantum convolution* that encodes a 2×2 image patch into a 4‑qubit
     state using Ry gates.
  2. A *random layer* of two‑qubit operations that expands the feature space.
  3. A *variational layer* (EstimatorQNN) that learns a mapping from the
     encoded state to a scalar output.

The circuit is built once in :meth:`__init__` and reused for every
forward pass.  The class is fully compatible with Qiskit’s
``EstimatorQNN`` interface and can be trained with any of the
available Qiskit primitives.

Typical usage::

    model = HybridQuanvolutionEstimator()
    # ``x`` is a batch of 2×2 patches flattened to shape (batch, 4)
    output = model(x)
"""

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
import torch

class HybridQuanvolutionEstimator:
    """Quantum neural network that combines a quanvolution filter and an
    EstimatorQNN head.
    """
    def __init__(self) -> None:
        # Input parameters: one per pixel in a 2×2 patch.
        self.input_params = [Parameter(f"x{i}") for i in range(4)]
        # Weight parameters for the variational layer.
        self.weight_params = [Parameter(f"w{i}") for i in range(4)]

        # Build the circuit.
        qc = QuantumCircuit(4)
        # Encoding: Ry gates for each pixel.
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)

        # Random layer: 8 two‑qubit gates (here we use a simple pattern).
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.cx(2, 3)
        qc.ry(0.5, 0)
        qc.rx(0.5, 1)
        qc.ry(0.5, 2)
        qc.rz(0.5, 3)

        # Variational layer: simple Ry rotations parameterised by weights.
        for i, p in enumerate(self.weight_params):
            qc.ry(p, i)

        # Observable: expectation value of Pauli‑Z on the first qubit.
        observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

        # Estimator primitive.
        estimator = StatevectorEstimator()

        # Wrap into EstimatorQNN.
        self.estimator_qnn = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            Tensor of shape (batch, 4) containing pixel values in the
            interval [0, π] (the Ry angles).  Values outside this range
            will be clipped.

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch,) with the expectation value for
            each input patch.
        """
        # Clamp input to [0, π] to stay within valid Ry range.
        inputs = torch.clamp(x, 0, torch.pi).numpy()
        # Qiskit EstimatorQNN expects a list of parameters per sample.
        param_dicts = [{p: val for p, val in zip(self.input_params, sample)}
                       for sample in inputs]
        # Run the estimator.
        results = self.estimator_qnn.run(param_dicts)
        # Convert to torch tensor.
        return torch.tensor(results, dtype=torch.float32)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

__all__ = ["HybridQuanvolutionEstimator"]
