from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple, Any
import numpy as np
import networkx as nx
import itertools

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN as QNN_SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info.operators.base_operator import BaseOperator
import qutip as qt


class HybridSamplerQNN:
    """
    Quantum counterpart of the hybrid sampler.
    Builds a parameterized ansatz with an auto‑encoder style swap‑test,
    wraps it in a Qiskit SamplerQNN, and provides evaluation and graph utilities.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        latent_dim: int = 3,
        trash_dim: int = 2,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim

        # Build ansatz
        self.ansatz = RealAmplitudes(num_qubits + trash_dim, reps=5)

        # Build auto‑encoder circuit
        self.circuit = self._build_autoencoder_circuit()

        # SamplerQNN wrapper
        self.sampler = Sampler()
        self.qnn = QNN_SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=2,
        )

    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.trash_dim + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent + trash with ansatz
        qc.compose(self.ansatz, range(0, self.latent_dim + self.trash_dim), inplace=True)
        qc.barrier()

        # Swap‑test auxiliary qubit
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        # Domain wall (optional perturbation)
        qc = self._domain_wall(qc, 0, 5)

        return qc

    @staticmethod
    def _domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        for i in range(a, b):
            circuit.x(i)
        return circuit

    def sample(self, latent_vector: np.ndarray) -> np.ndarray:
        """Run the quantum sampler on a latent vector."""
        # Pad latent_vector to match circuit parameter count
        params = np.array(latent_vector, dtype=np.float64)
        result = self.qnn.sample(params)
        return result

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = qt.Statevector.from_instruction(self._bind_circuit(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind_circuit(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self.circuit.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.circuit.parameters, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    @staticmethod
    def _state_fidelity(a: qt.Qobj, b: qt.Qobj) -> float:
        """Return the absolute squared overlap between pure states."""
        return abs((a.dag() * b)[0, 0]) ** 2

    def fidelity_adjacency(
        self,
        states: Sequence[qt.Qobj],
        threshold: float,
        *,
        secondary: float | None = None,
        secondary_weight: float = 0.5,
    ) -> nx.Graph:
        """Create a weighted adjacency graph from state fidelities."""
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, state_i), (j, state_j) in itertools.combinations(enumerate(states), 2):
            fid = self._state_fidelity(state_i, state_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph


def SamplerQNN(
    num_qubits: int = 2,
    *,
    latent_dim: int = 3,
    trash_dim: int = 2,
) -> HybridSamplerQNN:
    """Factory mirroring the classical helper returning a configured quantum network."""
    return HybridSamplerQNN(num_qubits, latent_dim=latent_dim, trash_dim=trash_dim)


__all__ = ["HybridSamplerQNN", "SamplerQNN"]
