from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute
from qiskit.circuit import Parameter, random_circuit
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, List, Sequence

# ------------------------------------------------------------------
# FastBaseEstimator adapted from the seed
# ------------------------------------------------------------------
class FastBaseEstimator:
    """
    Evaluates expectation values of a parameterised circuit for
    multiple parameter sets and observables.
    """

    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self, observables: Iterable[BaseOperator], parameter_sets: Sequence[Sequence[float]]
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results


# ------------------------------------------------------------------
# Hybrid quantum ConvGen251
# ------------------------------------------------------------------
class ConvGen251:
    """
    Quantum branch: two‑stage pipeline – a quanvolution circuit that
    encodes 2×2 image patches into qubits, followed by a
    self‑attention‑style circuit that weighs the resulting probabilities.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        backend=None,
        shots: int = 1024,
        embed_dim: int = 4,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.embed_dim = embed_dim

        # Quanvolution circuit
        self.n_qubits = kernel_size ** 2
        self.conv_circ = self._build_conv_circuit()

        # Self‑attention circuit
        self.attn_circ = self._build_attn_circuit()

        # Estimators for rapid batched evaluation
        self.conv_est = FastBaseEstimator(self.conv_circ)
        self.attn_est = FastBaseEstimator(self.attn_circ)

    def _build_conv_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            circ.rx(self.theta[i], i)
        circ.barrier()
        circ += random_circuit(self.n_qubits, 2)
        circ.measure_all()
        return circ

    def _build_attn_circuit(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.embed_dim)
        self.rot_params = [Parameter(f"r{i}") for i in range(3 * self.embed_dim)]
        self.ent_params = [Parameter(f"e{i}") for i in range(self.embed_dim - 1)]
        for i in range(self.embed_dim):
            circ.rx(self.rot_params[3 * i], i)
            circ.ry(self.rot_params[3 * i + 1], i)
            circ.rz(self.rot_params[3 * i + 2], i)
        for i in range(self.embed_dim - 1):
            circ.crx(self.ent_params[i], i, i + 1)
        circ.measure_all()
        return circ

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def run(self, data: np.ndarray) -> float:
        """
        Args:
            data: 2‑D array of shape (kernel_size, kernel_size)
        Returns:
            Scalar feature – weighted probability from the attention circuit.
        """
        # Encode patch into rotation angles
        flat = data.reshape(self.n_qubits)
        param_bind = {self.theta[i]: np.pi if v > self.threshold else 0 for i, v in enumerate(flat)}
        job = execute(self.conv_circ, self.backend,
                      shots=self.shots, parameter_binds=[param_bind])
        counts = job.result().get_counts()
        # Compute mean probability of measuring |1>
        prob1 = sum(int(state.count("1")) * count for state, count in counts.items())
        prob1 /= self.shots * self.n_qubits

        # Attention: use prob1 as a single scaling angle for all rotation params
        angle = np.pi * prob1
        rot_vals = [angle] * (3 * self.embed_dim)
        ent_vals = [0.0] * (self.embed_dim - 1)
        param_bind_attn = dict(zip(self.rot_params, rot_vals))
        param_bind_attn.update(dict(zip(self.ent_params, ent_vals)))
        job2 = execute(self.attn_circ, self.backend,
                       shots=self.shots, parameter_binds=[param_bind_attn])
        counts2 = job2.result().get_counts()
        # Extract average weight from measurement outcomes
        weight = sum(int(state[i]) * count for state, count in counts2
                     for i in range(self.embed_dim))
        weight /= self.shots * self.embed_dim
        return float(prob1 * weight)

    # ------------------------------------------------------------------
    # FastEstimator style evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Batch‑evaluate the quantum pipeline for a list of
        parameter sets.  Each parameter set should contain
        `self.n_qubits` angles for the conv circuit followed by
        `3*embed_dim + embed_dim-1` angles for the attention circuit.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            # Split parameters
            conv_vals = params[:self.n_qubits]
            attn_vals = params[self.n_qubits:]
            # Conv expectation
            conv_exp = self.conv_est.evaluate(
                [Statevector.from_instruction(self.conv_circ)],
                [conv_vals]
            )[0][0]
            # Attention expectation
            attn_exp = self.attn_est.evaluate(
                [Statevector.from_instruction(self.attn_circ)],
                [attn_vals]
            )[0][0]
            results.append([conv_exp, attn_exp])
        return results


__all__ = ["ConvGen251", "FastBaseEstimator"]
