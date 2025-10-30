import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from typing import Iterable, Sequence, List
from dataclasses import dataclass

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

def _apply_layer(circuit: QuantumCircuit, params: FraudLayerParameters, clip: bool) -> None:
    """Apply a photonic‑style layer to a quantum circuit, clipping parameters if requested."""
    # Placeholder for a beam‑splitter; in practice a CX or RZ can emulate a BS.
    circuit.h(0)
    for i, phase in enumerate(params.phases):
        circuit.rz(phase, i)
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        circuit.rz(_clip(r, 5) if clip else r, i)
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        circuit.rx(_clip(r, 5) if clip else r, i)
    for i, k in enumerate(params.kerr):
        circuit.rz(_clip(k, 1) if clip else k, i)

def _conv_layer(circuit: QuantumCircuit, qubits: Sequence[int], params: np.ndarray) -> None:
    """A 2‑qubit convolution block used in the QCNN‑style quantum layers."""
    q1, q2 = qubits
    circuit.cx(q1, q2)
    circuit.rz(params[0], q1)
    circuit.ry(params[1], q2)
    circuit.cx(q2, q1)

def _pool_layer(circuit: QuantumCircuit, qubits: Sequence[int], params: np.ndarray) -> None:
    """A 2‑qubit pooling block that entangles and measures."""
    q1, q2 = qubits
    circuit.cx(q1, q2)
    circuit.rz(params[0], q1)
    circuit.ry(params[1], q2)
    circuit.measure(q1, q2)

class SelfAttentionHybrid:
    """Quantum variational self‑attention circuit that incorporates QCNN‑style
    convolution and pooling layers and uses fraud‑style parameter clipping.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(self,
                       rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       qc_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Rotation layer
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entanglement
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        # QCNN convolution & pooling
        idx = 0
        for layer in range(self.n_layers):
            # Convolution on pairs (0,1),(2,3),...
            for q in range(0, self.n_qubits - 1, 2):
                _conv_layer(circuit, (q, q + 1), qc_params[idx:idx + 3])
                idx += 3
            # Pooling on adjacent pairs
            for q in range(0, self.n_qubits - 1, 2):
                _pool_layer(circuit, (q, q + 1), qc_params[idx:idx + 2])
                idx += 2
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            qc_params: np.ndarray,
            shots: int = 1024) -> dict:
        circuit = self._build_circuit(rotation_params, entangle_params, qc_params)
        job = qiskit.execute(circuit, backend, shots=shots)
        return job.result().get_counts(circuit)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        """Expectation‑value evaluation using Statevector for each parameter set."""
        results: List[List[complex]] = []
        for params in parameter_sets:
            rot = np.array(params[:self.n_qubits * 3])
            ent = np.array(params[self.n_qubits * 3: self.n_qubits * 3 + (self.n_qubits - 1)])
            qc = np.array(params[self.n_qubits * 3 + (self.n_qubits - 1):])
            circuit = self._build_circuit(rot, ent, qc)
            state = Statevector.from_instruction(circuit)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

def SelfAttention() -> SelfAttentionHybrid:
    """Factory that returns a pre‑configured quantum self‑attention circuit."""
    n_qubits = 4
    n_layers = 2
    backend = qiskit.Aer.get_backend("qasm_simulator")
    return SelfAttentionHybrid(n_qubits=n_qubits, n_layers=n_layers)

__all__ = ["SelfAttentionHybrid", "SelfAttention"]
