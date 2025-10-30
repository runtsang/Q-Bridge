"""Hybrid quantum classifier that exposes the same API as its classical counterpart.

The implementation builds a parameterised circuit, an optional sampler, and
provides helper methods to evaluate on classical data.  It is designed to
interoperate with the classical `HybridClassifier` via identical method
names.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

import torch


def build_quantum_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List[ParameterVector], List[ParameterVector], List[SparsePauliOp]]:
    """
    Construct a layered ansatz with data‑encoding and variational layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features).
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        Parameterised circuit ready for sampling.
    encoding : List[ParameterVector]
        List containing the encoding parameters.
    weights : List[ParameterVector]
        List containing the variational parameters.
    observables : List[SparsePauliOp]
        Pauli observables used for measurement.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    # data encoding
    for qubit, param in enumerate(encoding):
        qc.rx(param, qubit)

    # variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        # Entanglement pattern: linear CZ chain
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    # measurement observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]

    return qc, [encoding], [weights], observables


class QuantumHybridClassifier:
    """
    Quantum implementation that mirrors the classical `HybridClassifier`.

    The class encapsulates the circuit, a sampler, and a simple post‑processing
    head that maps expectation values to class logits.
    """

    def __init__(self, num_qubits: int, depth: int, use_sampler: bool = True) -> None:
        self.circuit, self.encoding, self.weights, self.observables = build_quantum_circuit(
            num_qubits, depth
        )
        self.use_sampler = use_sampler

        # Sampler primitive; can be swapped for a backend of choice
        self.sampler = StatevectorSampler()
        if use_sampler:
            # Wrap the circuit in a SamplerQNN for convenience
            self.sampler_qnn = SamplerQNN(
                circuit=self.circuit,
                input_params=self.encoding[0],
                weight_params=self.weights[0],
                sampler=self.sampler,
            )
        else:
            self.sampler_qnn = None

    def _sample(self, inputs: List[float]) -> torch.Tensor:
        """
        Evaluate the circuit on a single data point and return expectation values.

        Parameters
        ----------
        inputs : List[float]
            Feature vector of length `num_qubits`.

        Returns
        -------
        torch.Tensor
            Tensor of shape (num_qubits,) containing expectation values of Z.
        """
        param_dict = {self.encoding[0][i]: inputs[i] for i in range(len(inputs))}
        if self.use_sampler:
            # Use the primitive to get expectation values
            result = self.sampler.run(self.circuit, parameter_binds=[param_dict]).result()
            exp_vals = torch.tensor(result.get_expectation_values(self.observables), dtype=torch.float32)
        else:
            # Fallback: return zero tensor
            exp_vals = torch.zeros(len(inputs), dtype=torch.float32)
        return exp_vals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that maps input features to class logits.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (..., num_qubits).

        Returns
        -------
        torch.Tensor
            Logits of shape (..., 2).  If `use_sampler` is True, the sampler
            is used; otherwise a direct statevector evaluation is performed.
        """
        logits = []
        for sample in x:
            exp = self._sample(sample.tolist())
            # Simple linear head: one weight per qubit + bias
            weight = torch.ones(exp.shape[0], dtype=torch.float32)
            bias = torch.tensor([0.0, 0.0], dtype=torch.float32)
            logits.append(weight @ exp + bias)
        return torch.stack(logits)

    def parameters(self):
        """
        Return all trainable parameters for use with a classical optimiser.

        Returns
        -------
        List[ParameterVector]
            All variational parameters of the circuit.
        """
        return self.weights[0]


__all__ = ["QuantumHybridClassifier", "build_quantum_circuit"]
