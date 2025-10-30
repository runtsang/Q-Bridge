"""Quantum autoencoder that uses a variational RealAmplitudes circuit, swap‑test, and fidelity graph analysis."""

from __future__ import annotations

import numpy as np
import networkx as nx

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, assemble
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

def _state_fidelity_vector(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(np.abs(np.vdot(a_norm, b_norm)) ** 2)

class HybridAutoEncoder:
    """Quantum autoencoder built from a parameterised RealAmplitudes ansatz and a swap‑test."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        shots: int = 1024,
        backend=None,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.backend = backend or AerSimulator()
        self.sampler = StatevectorSampler(self.backend)
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Domain‑wall style pre‑rotation on the first half of the qubits
        for i in range(self.num_latent + self.num_trash):
            circuit.x(i)

        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        circuit.barrier()

        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def forward(self, params: np.ndarray) -> np.ndarray:
        """Evaluate the circuit with the given parameter vector and return the measurement expectation."""
        return np.array(self.qnn(params))

    def analyze_latent_fidelity(
        self,
        param_vectors: np.ndarray,
        threshold: float = 0.8,
    ) -> nx.Graph:
        """Build a graph where nodes are input parameter sets and edges connect statevectors with high fidelity."""
        statevectors = [self.sampler.run(self.circuit, [params]).statevector for params in param_vectors]
        graph = nx.Graph()
        graph.add_nodes_from(range(len(statevectors)))
        for i in range(len(statevectors)):
            for j in range(i + 1, len(statevectors)):
                fid = _state_fidelity_vector(statevectors[i], statevectors[j])
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
        return graph

def HybridAutoEncoderFactory(
    num_latent: int = 3,
    num_trash: int = 2,
    shots: int = 1024,
    backend=None,
) -> HybridAutoEncoder:
    return HybridAutoEncoder(num_latent, num_trash, shots, backend)

def train_qml_autoencoder(
    model: HybridAutoEncoder,
    data: np.ndarray,
    *,
    maxfun: int = 200,
    tolerance: float = 1e-3,
) -> np.ndarray:
    """Simple COBYLA optimisation of the circuit parameters to minimise MSE between output and input."""
    def loss(params):
        preds = model.forward(params)
        return np.mean((preds - data) ** 2)

    optimiser = COBYLA()
    result = optimiser.optimize(
        num_vars=len(model.circuit.parameters),
        objective_function=loss,
        initial_point=np.zeros(len(model.circuit.parameters)),
        maxfun=maxfun,
        tolerance=tolerance,
    )
    model.circuit.assign_parameters(result.optimal_point)
    return result.optimal_point

__all__ = [
    "HybridAutoEncoder",
    "HybridAutoEncoderFactory",
    "train_qml_autoencoder",
]
