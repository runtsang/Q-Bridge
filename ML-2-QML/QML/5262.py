import itertools
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Statevector, Unitary
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import networkx as nx
from dataclasses import dataclass
from typing import Sequence, List, Tuple, Iterable

class GraphQNNHybrid:
    """Hybrid Graph QNN class with quantum backend.

    Mirrors the classical implementation but replaces linear layers with
    variational unitaries and uses quantum state fidelities for graph
    construction.  The class provides factory methods for a quantum
    convolution (quanvolution) filter and a photonic fraud‑detection
    program, enabling end‑to‑end quantum experiments.
    """

    def __init__(self, arch: Sequence[int]) -> None:
        self.arch = list(arch)

    def random_network(self, samples: int) -> Tuple[List[List[Unitary]], List[Tuple[Statevector, Statevector]], Statevector]:
        target = Statevector.random(self.arch[-1])
        training = [(Statevector.random(self.arch[-1]), target) for _ in range(samples)]
        unitaries: List[List[Unitary]] = [[]]
        for layer in range(1, len(self.arch)):
            ops = [Unitary.random(self.arch[layer-1] + 1) for _ in range(self.arch[layer])]
            unitaries.append(ops)
        return unitaries, training, target

    def feedforward(self, unitaries: Sequence[Sequence[Unitary]], samples: Iterable[Tuple[Statevector, Statevector]]) -> List[List[Statevector]]:
        outputs = []
        for inp, _ in samples:
            states = [inp]
            current = inp
            for layer in range(1, len(self.arch)):
                op = unitaries[layer][0]
                for gate in unitaries[layer][1:]:
                    op = op @ gate
                current = op @ current
                states.append(current)
            outputs.append(states)
        return outputs

    @staticmethod
    def state_fidelity(a: Statevector, b: Statevector) -> float:
        return abs((a.data.conj().T @ b.data)[0]) ** 2

    @staticmethod
    def fidelity_adjacency(states: Sequence[Statevector], threshold: float, *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(states)))
        for (i, s_i), (j, s_j) in itertools.combinations(enumerate(states), 2):
            fid = GraphQNNHybrid.state_fidelity(s_i, s_j)
            if fid >= threshold:
                graph.add_edge(i, j, weight=1.0)
            elif secondary is not None and fid >= secondary:
                graph.add_edge(i, j, weight=secondary_weight)
        return graph

    @staticmethod
    def Conv(kernel_size: int = 2, backend: qiskit.providers.BaseBackend | None = None, shots: int = 100, threshold: float = 0.0):
        """Return a quanvolution circuit that evaluates a 2‑D kernel."""
        class QuanvCircuit:
            def __init__(self):
                self.n_qubits = kernel_size ** 2
                self.circuit = qiskit.QuantumCircuit(self.n_qubits)
                self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
                for i in range(self.n_qubits):
                    self.circuit.rx(self.theta[i], i)
                self.circuit.barrier()
                self.circuit += random_circuit(self.n_qubits, 2)
                self.circuit.measure_all()
                self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
                self.shots = shots
                self.threshold = threshold

            def run(self, data: np.ndarray) -> float:
                data = np.reshape(data, (1, self.n_qubits))
                param_binds = []
                for datum in data:
                    bind = {theta: np.pi if val > self.threshold else 0 for theta, val in zip(self.theta, datum)}
                    param_binds.append(bind)
                job = qiskit.execute(self.circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
                result = job.result().get_counts(self.circuit)
                counts = 0
                for bits, cnt in result.items():
                    counts += sum(int(b) for b in bits) * cnt
                return counts / (self.shots * self.n_qubits)
        return QuanvCircuit()

    @staticmethod
    def FraudDetection(input_params, layers):
        """Build a Strawberry Fields program mirroring the photonic fraud‑detection model."""
        @dataclass
        class FraudLayerParameters:
            bs_theta: float
            bs_phi: float
            phases: tuple[float, float]
            squeeze_r: tuple[float, float]
            squeeze_phi: tuple[float, float]
            displacement_r: tuple[float, float]
            displacement_phi: tuple[float, float]
            kerr: tuple[float, float]

        def _clip(value: float, bound: float) -> float:
            return max(-bound, min(bound, value))

        program = sf.Program(2)
        with program.context as q:
            def _apply_layer(modes, params: FraudLayerParameters, clip: bool):
                BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
                for i, phase in enumerate(params.phases):
                    Rgate(phase) | modes[i]
                for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                    Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
                BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
                for i, phase in enumerate(params.phases):
                    Rgate(phase) | modes[i]
                for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                    Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
                for i, k in enumerate(params.kerr):
                    Kgate(k if not clip else _clip(k, 1)) | modes[i]
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

__all__ = ["GraphQNNHybrid"]
