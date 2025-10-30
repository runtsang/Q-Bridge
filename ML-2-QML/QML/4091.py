from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Tuple, Optional

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import Pauli

class HybridFastEstimator:
    """Quantum estimator for parametrized circuits and hybrid CNN‑quantum kernels.

    Parameters
    ----------
    circuit : QuantumCircuit | Tuple[object, QuantumCircuit]
        If a tuple (cnn, circuit) is passed, ``cnn`` is a torch.nn.Module that
        extracts features from the input parameters and the circuit operates on
        those features.  Otherwise, ``circuit`` is a pure quantum circuit.
    shots : int | None, optional
        Number of shots to use for sampling; if None, state‑vector simulation
        is used.
    backend : str, optional
        Name of the Aer backend to use for sampling. Default is ``aer_simulator``.
    """

    def __init__(
        self,
        circuit: QuantumCircuit | Tuple[object, QuantumCircuit],
        *,
        shots: Optional[int] = None,
        backend: str = "aer_simulator",
    ) -> None:
        if isinstance(circuit, tuple):
            self.cnn, self.quantum_circuit = circuit
            self.is_hybrid = True
        else:
            self.quantum_circuit = circuit
            self.cnn = None
            self.is_hybrid = False

        self.shots = shots
        self.backend = backend
        self.simulator = Aer.get_backend(backend)

        self.parameters = list(self.quantum_circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.parameters, parameter_values))
        return self.quantum_circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Pauli],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        results: List[List[complex]] = []

        for values in parameter_sets:
            if self.is_hybrid:
                import torch
                torch_input = torch.as_tensor(values, dtype=torch.float32).unsqueeze(0)
                features = self.cnn(torch_input).detach().cpu().numpy().flatten()
                bound_circuit = self._bind(features)
            else:
                bound_circuit = self._bind(values)

            if self.shots is None:
                state = Statevector.from_instruction(bound_circuit)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                job = execute(bound_circuit, self.simulator, shots=self.shots)
                result = job.result()
                counts = result.get_counts(bound_circuit)
                row = [self._counts_to_expectation(counts, obs) for obs in observables]
            results.append(row)

        return results

    def _counts_to_expectation(self, counts: dict, pauli: Pauli) -> complex:
        # Simplified conversion: only works for Z observables
        if pauli.to_label().replace('I', '')!= '':
            raise NotImplementedError("Only Z observables are supported in shot mode.")
        exp_val = 0.0
        for bitstring, freq in counts.items():
            eigen = 1
            for qubit, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    eigen *= -1
            exp_val += eigen * freq
        return exp_val / sum(counts.values())

    @staticmethod
    def build_random_circuit(num_qubits: int, num_ops: int = 50) -> QuantumCircuit:
        """Construct a parameter‑efficient variational circuit.

        The circuit consists of a random layer of single‑qubit rotations
        followed by a chain of RX/RY/RZ gates and a CNOT ladder.
        """
        qc = QuantumCircuit(num_qubits)
        for _ in range(num_ops):
            target = np.random.randint(num_qubits)
            gate = np.random.choice(['rx', 'ry', 'rz'])
            theta = np.random.uniform(0, 2*np.pi)
            getattr(qc, gate)(target, theta)
        for q in range(num_qubits):
            qc.rx(0.0, q)
            qc.ry(0.0, q)
            qc.rz(0.0, q)
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        return qc

__all__: list[str] = ["HybridFastEstimator"]
