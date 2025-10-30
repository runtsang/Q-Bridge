"""Quantum counterpart of CombinedEstimatorQNN.

The quantum module mirrors the classical API but replaces the regression,
kernel, graph, and autoencoder with quantum‑equivalent constructs:

* `QuantumRegression` – a variational EstimatorQNN that maps two parameters to a single
  expectation value.
* `QuantumKernel` – a TorchQuantum module that evaluates a fixed RY‑encoding ansatz.
* `QuantumGraph` – produces a graph from state fidelities of a small variational circuit.
* `QuantumAutoencoder` – a SamplerQNN built from a RealAmplitudes ansatz and a swap‑test
  encoder/decoder.

All components are assembled inside `CombinedEstimatorQNNQuantum`, exposing the same
method names (`predict`, `gram_matrix`, `build_graph`, `encode`, `decode`) so that
classical and quantum code can be swapped with minimal friction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.utils import algorithm_globals
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


# --------------------------------------------------------------------------- #
# 1. Variational regression (from EstimatorQNN)
# --------------------------------------------------------------------------- #
@dataclass
class QuantumRegression:
    """Variational estimator that returns a single expectation value."""

    def __post_init__(self):
        params = [Parameter(f"rx_{i}") for i in range(2)]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)

        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        self.model = EstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[params[0]],
            weight_params=[params[1]],
            estimator=estimator,
        )

    def predict(self, input_val: float, weight: float) -> float:
        """Return the expectation value for a single data point."""
        return float(self.model.predict([input_val], [weight])[0])


# --------------------------------------------------------------------------- #
# 2. TorchQuantum kernel (from QuantumKernelMethod)
# --------------------------------------------------------------------------- #
class TorchQuantumKernel(tq.QuantumModule):
    """Fixed RY‑encoding ansatz used to compute a quantum kernel."""

    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.ansatz = tq.QuantumModule()
        # Build a list of operations
        self.func_list = [
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ]

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the absolute overlap of the two encoded states."""
        self.forward(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def quantum_gram_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Compute Gram matrix using the TorchQuantum kernel."""
    kernel = TorchQuantumKernel()
    return np.array([[kernel.evaluate(x, y).item() for y in b] for x in a])


# --------------------------------------------------------------------------- #
# 3. Quantum graph (from GraphQNN)
# --------------------------------------------------------------------------- #
def _random_qunitary(num_qubits: int) -> Statevector:
    """Generate a random unitary as a Statevector."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    mat = np.linalg.qr(mat)[0]
    return Statevector(mat)


def quantum_random_network(qnn_arch: Sequence[int], samples: int) -> Tuple[Sequence[int], List[Statevector], List[Tuple[Statevector, Statevector]], Statevector]:
    """Create a random quantum network and training data."""
    target_unitary = _random_qunitary(qnn_arch[-1])
    training_data = [(Statevector.random(qnn_arch[-1]), target_unitary * Statevector.random(qnn_arch[-1])) for _ in range(samples)]
    return list(qnn_arch), [target_unitary], training_data, target_unitary


def quantum_fidelity(a: Statevector, b: Statevector) -> float:
    """Squared overlap of two pure states."""
    return abs((a.dag() @ b)[0, 0]) ** 2


def quantum_fidelity_adjacency(
    states: Sequence[Statevector],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(len(states)))
    for (i, ai), (j, bj) in itertools.combinations(enumerate(states), 2):
        fid = quantum_fidelity(ai, bj)
        if fid >= threshold:
            graph.add_edge(i, j, weight=1.0)
        elif secondary is not None and fid >= secondary:
            graph.add_edge(i, j, weight=secondary_weight)
    return graph


# --------------------------------------------------------------------------- #
# 4. Quantum autoencoder (from Autoencoder)
# --------------------------------------------------------------------------- #
def quantum_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Build a swap‑test autoencoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational part
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test
    qc.h(qr[num_latent + 2 * num_trash])
    for i in range(num_trash):
        qc.cswap(qr[num_latent + 2 * num_trash], qr[num_latent + i], qr[num_latent + num_trash + i])
    qc.h(qr[num_latent + 2 * num_trash])
    qc.measure(qr[num_latent + 2 * num_trash], cr[0])
    return qc


def quantum_autoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a SamplerQNN that implements a quantum autoencoder."""
    qc = quantum_autoencoder_circuit(num_latent, num_trash)
    sampler = Sampler()
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# 5. Public hybrid quantum interface
# --------------------------------------------------------------------------- #
class CombinedEstimatorQNNQuantum:
    """Quantum analogue of CombinedEstimatorQNN."""

    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        self.regression = QuantumRegression()
        self.kernel = TorchQuantumKernel()
        self.autoencoder = quantum_autoencoder(num_latent, num_trash)
        self.graph = {
            "random_network": quantum_random_network,
            "fidelity_adjacency": quantum_fidelity_adjacency,
        }

    # --------------------------------------------------------------------- #
    # 5.1 Regression
    # --------------------------------------------------------------------- #
    def predict(self, x: float, w: float) -> float:
        return self.regression.predict(x, w)

    # --------------------------------------------------------------------- #
    # 5.2 Kernel
    # --------------------------------------------------------------------- #
    def gram_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return quantum_gram_matrix(a, b)

    # --------------------------------------------------------------------- #
    # 5.3 Graph
    # --------------------------------------------------------------------- #
    def build_graph(self, states: Sequence[Statevector], threshold: float) -> nx.Graph:
        return quantum_fidelity_adjacency(states, threshold)

    # --------------------------------------------------------------------- #
    # 5.4 Autoencoder
    # --------------------------------------------------------------------- #
    def encode(self, sample: Statevector) -> Statevector:
        """Return the encoded state by sampling the autoencoder."""
        probs = self.autoencoder.sample(sample.data)
        # Convert probability distribution to a pure state (approximate)
        state = Statevector(probs)
        return state

    def decode(self, latent: Statevector) -> Statevector:
        """Decode a latent state back to the original space."""
        return latent  # placeholder – full decoding would require inverse circuit

    def train_autoencoder(self, data: List[Statevector], *, epochs: int = 10) -> None:
        """Simplified training loop using COBYLA."""
        optimizer = COBYLA()
        for _ in range(epochs):
            for sample in data:
                # Objective: minimize distance to original state
                def objective(params):
                    self.autoencoder.weight_params = params
                    encoded = self.encode(sample)
                    return 1.0 - quantum_fidelity(sample, encoded)
                optimizer.minimize(objective, self.autoencoder.weight_params)

__all__ = ["CombinedEstimatorQNNQuantum"]
