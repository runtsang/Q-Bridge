from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List, Tuple

import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import networkx as nx


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer with an optional quanvolution filter."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    conv_kernel: int = 2  # size of the quantum convolution filter


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program that mirrors the photonic architecture."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program


class QuanvCircuit:
    """Quantum convolution filter that emulates a classical CNN kernel."""
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


def build_quantum_conv(kernel_size: int, threshold: float = 0.5) -> QuanvCircuit:
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return QuanvCircuit(kernel_size, backend, shots=200, threshold=threshold)


class FraudDetectionHybrid:
    """Hybrid quantum‑classical fraud‑detection model."""
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.program = build_fraud_detection_program(input_params, layers)
        self.quantum_conv = build_quantum_conv(kernel_size=input_params.conv_kernel, threshold=0.5)

    def run(self, data: np.ndarray) -> float:
        """Process classical data through a quanvolution filter and a photonic circuit."""
        # Quantum convolution on the raw image
        conv_out = self.quantum_conv.run(data)
        # Encode the convolution output as a displacement on mode 0
        with sf.Program(2) as prog:
            Dgate(conv_out, 0) | 0
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
        result = eng.run(self.program)
        probs = result.state.probability([0, 0])
        return probs[0]

    def fidelity_adjacency(self, samples: List[tuple[np.ndarray, np.ndarray]], threshold: float,
                           *, secondary: float | None = None, secondary_weight: float = 0.5) -> nx.Graph:
        """Build a weighted graph from fidelities between output states of the program."""
        def state_fidelity(a: sf.State, b: sf.State) -> float:
            # Absolute squared overlap between pure states
            return abs((a.dag() * b)[0, 0]) ** 2

        graph = nx.Graph()
        outputs: List[sf.State] = []
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 8})
        for data, _ in samples:
            conv_out = self.quantum_conv.run(data)
            with sf.Program(2) as prog:
                Dgate(conv_out, 0) | 0
            out_state = eng.run(self.program).state
            outputs.append(out_state)
        for i, a in enumerate(outputs):
            for j, b in enumerate(outputs[i+1:], start=i+1):
                fid = state_fidelity(a, b)
                if fid >= threshold:
                    graph.add_edge(i, j, weight=1.0)
                elif secondary is not None and fid >= secondary:
                    graph.add_edge(i, j, weight=secondary_weight)
        return graph
