from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN, EstimatorQNN as QEstimatorQNN
from qiskit.primitives import Sampler, Estimator
import torch
import numpy as np

class UnifiedQNN:
    """
    Quantum‑centric counterpart of the classical UnifiedQNN.
    It builds a variational circuit that mirrors the SamplerQNN and EstimatorQNN
    structures and uses Qiskit’s primitives to execute the forward pass.
    """

    def __init__(self,
                 mode: str = "sampler",
                 input_dim: int = 2,
                 quantum_dim: int = 2) -> None:
        """
        Parameters
        ----------
        mode : {"sampler", "estimator"}
            Selects the quantum backend analogous to the classical mode.
        input_dim : int
            Number of classical input features (used to size ParameterVector).
        quantum_dim : int
            Number of trainable weight parameters in the circuit.
        """
        self.mode = mode
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim

        if mode == "sampler":
            self._build_sampler()
        elif mode == "estimator":
            self._build_estimator()
        else:
            raise ValueError("mode must be'sampler' or 'estimator'")

    def _build_sampler(self) -> None:
        """Constructs a SamplerQNN variational circuit."""
        inputs = ParameterVector("input", self.input_dim)
        weights = ParameterVector("weight", self.quantum_dim)

        qc = QuantumCircuit(self.input_dim)
        # Encode inputs
        for i, p in enumerate(inputs):
            qc.ry(p, i)
        # Parameterized rotations
        for i, p in enumerate(weights):
            qc.ry(p, i)
        # Entangling layer
        qc.cx(0, 1)

        self.sampler = Sampler()
        self.model = QSamplerQNN(
            circuit=qc,
            input_params=inputs,
            weight_params=weights,
            sampler=self.sampler
        )

    def _build_estimator(self) -> None:
        """Constructs an EstimatorQNN variational circuit."""
        inputs = ParameterVector("input", self.input_dim)
        weights = ParameterVector("weight", self.quantum_dim)

        qc = QuantumCircuit(self.input_dim)
        for i, p in enumerate(inputs):
            qc.ry(p, i)
        for i, p in enumerate(weights):
            qc.ry(p, i)

        # For simplicity, we use the Pauli‑Y observable on the first qubit
        from qiskit.quantum_info import SparsePauliOp
        observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])

        self.estimator = Estimator()
        self.model = QEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=inputs,
            weight_params=weights,
            estimator=self.estimator
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the quantum circuit with the given classical inputs.

        Args:
            x: Tensor of shape (..., input_dim)

        Returns:
            Tensor of probabilities (sampler) or expectation values (estimator).
        """
        # Prepare the mapping from ParameterVector names to input values
        mapping = {}
        for idx, val in enumerate(x.squeeze().tolist()):
            mapping[f"input{idx}"] = val

        if self.mode == "sampler":
            probs = self.model.forward(mapping)
            return torch.tensor(probs, dtype=torch.float32)
        else:
            exp_vals = self.model.forward(mapping)
            return torch.tensor(exp_vals, dtype=torch.float32)

__all__ = ["UnifiedQNN"]
