from __future__ import annotations
from typing import Iterable, Tuple

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN


class HybridSamplerClassifier:
    """
    Quantum sampler‑classifier that mirrors the classical HybridSamplerClassifier.

    * Encoding: 2‑dimensional classical feature vector mapped to qubits via RX gates.
    * Variational ansatz: depth‑controlled layers of Ry rotations and CZ entangling gates.
    * Sampling: StatevectorSampler used to obtain measurement statistics; first qubit
      interpreted as the binary class label.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        depth: int = 2,
        device: str | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit, self.input_params, self.weight_params, self.observables = self._build_circuit()
        self.sampler = StatevectorSampler(device=device)
        self.sampler_qnn = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> Tuple[QuantumCircuit, Iterable[ParameterVector], Iterable[ParameterVector], list[SparsePauliOp]]:
        """
        Build a layered ansatz with explicit encoding and variational parameters.
        Mirrors build_classifier_circuit from the reference.
        """
        encoding = ParameterVector("x", self.num_qubits)
        weights = ParameterVector("theta", self.num_qubits * self.depth)

        qc = QuantumCircuit(self.num_qubits)
        for param, qubit in zip(encoding, range(self.num_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(self.depth):
            for qubit in range(self.num_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(self.num_qubits - 1):
                qc.cz(qubit, qubit + 1)

        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (self.num_qubits - i - 1))
            for i in range(self.num_qubits)
        ]
        return qc, list(encoding), list(weights), observables

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: accepts a batch of 2‑dimensional classical feature vectors,
        binds them to the encoding parameters, runs the sampler, and returns
        a probability distribution over two outcomes.
        """
        if x.shape[-1]!= self.num_qubits:
            raise ValueError(f"Expected input dimension {self.num_qubits}, got {x.shape[-1]}")

        probs = []
        for vec in x:
            # Bind encoding parameters
            param_dict = {str(p): float(vec[i].item()) for i, p in enumerate(self.input_params)}
            # Sample 1024 shots
            samples = self.sampler.sample(self.circuit, param_dict, shots=1024)
            # Convert to torch tensor
            samples_t = torch.tensor(samples[:, 0], dtype=torch.float32, device=x.device)
            prob0 = samples_t.mean().item()
            probs.append(torch.tensor([prob0, 1 - prob0], device=x.device))
        return torch.stack(probs)


__all__ = ["HybridSamplerClassifier"]
