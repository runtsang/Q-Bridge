from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import numpy as np
import torch
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --- Photonic‑style layer parameters (kept for compatibility) ---
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

def _apply_photonic_layer(q: Sequence, params: FraudLayerParameters, clip: bool = True) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | q[i]
    BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | q[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | q[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | q[i]

def build_photonic_fraud_program(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters]) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_photonic_layer(q, layer, clip=True)
    return prog

# --- QCNN ansatz using Qiskit ---
def conv_circuit(params: Sequence[float]) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    qc.cx(1, 0)
    qc.rz(np.pi / 2, 0)
    return qc

def conv_layer(num_qubits: int, param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(0, num_qubits, 2):
        qc.append(conv_circuit(params[i:i+3]), [i, i+1])
    for i in range(1, num_qubits-1, 2):
        qc.append(conv_circuit(params[i:i+3]), [i, i+1])
    return qc

def pool_circuit(params: Sequence[float]) -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.rz(-np.pi / 2, 1)
    qc.cx(1, 0)
    qc.rz(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 1)
    return qc

def pool_layer(num_qubits: int, sources: List[int], sinks: List[int], param_prefix: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    params = ParameterVector(param_prefix, length=len(sources) * 3)
    for src, snk in zip(sources, sinks):
        qc.append(pool_circuit(params[:3]), [src, snk])
        params = params[3:]
    return qc

def build_qcnn_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    qc.compose(conv_layer(num_qubits, "c1"), inplace=True)
    qc.compose(pool_layer(num_qubits, list(range(num_qubits//2)),
                          list(range(num_qubits//2, num_qubits)), "p1"), inplace=True)
    for d in range(1, depth):
        qc.compose(conv_layer(num_qubits, f"c{d+1}"), inplace=True)
        qc.compose(pool_layer(num_qubits, list(range(num_qubits//(2**(d+1)))),
                              list(range(num_qubits//(2**(d+1)), num_qubits//(2**d))), f"p{d+1}"), inplace=True)
    return qc

# --- Quantum kernel via TorchQuantum ---
class QuantumKernel:
    """Fixed ansatz computing overlap of two encoded states."""
    def __init__(self) -> None:
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            self.q_device.apply(tq.RY, wires=[i], params=x[:, i])
        for i in range(self.n_wires):
            self.q_device.apply(tq.RY, wires=[i], params=-y[:, i])
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    kernel = QuantumKernel()
    return np.array([[kernel.forward(x, y).item() for y in b] for x in a])

# --- Classical classifier analogue for compatibility ---
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    circuit = QuantumCircuit(num_qubits)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)
    idx = 0
    for _ in range(depth):
        for i in range(num_qubits):
            circuit.ry(weights[idx], i)
            idx += 1
        for i in range(num_qubits - 1):
            circuit.cz(i, i + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --- Hybrid quantum factory ---
class FraudDetectionHybrid:
    """
    Quantum counterpart of :class:`FraudDetectionHybrid` (classical).  Builds a
    photonic fraud circuit, a QCNN ansatz, a TorchQuantum kernel, and a
    variational classifier circuit.  The public API mirrors the classical
    implementation to enable side‑by‑side experimentation.
    """
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 qcnn_depth: int = 3,
                 num_qubits: int = 8,
                 classifier_depth: int = 2) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.qcnn_depth = qcnn_depth
        self.num_qubits = num_qubits
        self.classifier_depth = classifier_depth

    def build_photonic_fraud(self) -> sf.Program:
        """Return the Strawberry Fields program for fraud detection."""
        return build_photonic_fraud_program(self.input_params, self.layers)

    def build_qcnn_ansatz(self) -> QuantumCircuit:
        """Return a QCNN ansatz circuit."""
        return build_qcnn_ansatz(self.num_qubits, self.qcnn_depth)

    def build_kernel(self) -> QuantumKernel:
        """Return the TorchQuantum kernel module."""
        return QuantumKernel()

    def build_classifier(self) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
        """Return the variational classifier circuit and metadata."""
        return build_classifier_circuit(self.num_qubits, self.classifier_depth)

__all__ = ["FraudLayerParameters",
           "build_photonic_fraud_program",
           "build_qcnn_ansatz",
           "QuantumKernel",
           "build_classifier_circuit",
           "FraudDetectionHybrid"]
