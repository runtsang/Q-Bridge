from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ------------------------------------------------------------------
# Quantum fraud‑detection layer
# ------------------------------------------------------------------
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

def _apply_layer(
    modes: Sequence, params: FraudLayerParameters, *, clip: bool
) -> None:
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
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# ------------------------------------------------------------------
# Quantum self‑attention
# ------------------------------------------------------------------
class QuantumSelfAttention:
    """Basic quantum circuit representing a self‑attention style block."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _build_circuit(
        self, rotation_params: np.ndarray, entangle_params: np.ndarray
    ) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(
        self,
        backend: qiskit.providers.BaseBackend,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        circuit = self._build_circuit(rotation_params, entangle_params)
        job = qiskit.execute(circuit, backend=backend, shots=shots)
        return job.result().get_counts(circuit)

# ------------------------------------------------------------------
# Quantum classifier ansatz
# ------------------------------------------------------------------
def build_classifier_circuit(
    num_qubits: int, depth: int
) -> Tuple[QuantumCircuit, Sequence[ParameterVector], Sequence[ParameterVector], Sequence[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables

# ------------------------------------------------------------------
# Quantum fully‑connected layer helper
# ------------------------------------------------------------------
class QuantumFullyConnectedLayer:
    """Parameterized quantum circuit that mimics a classical fully‑connected layer."""
    def __init__(self, n_qubits: int, backend: qiskit.providers.BaseBackend, shots: int = 1024):
        self._circuit = QuantumCircuit(n_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self._circuit.h(range(n_qubits))
        self._circuit.barrier()
        self._circuit.ry(self.theta, range(n_qubits))
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        result = job.result().get_counts(self._circuit)
        counts = np.array(list(result.values()))
        states = np.array([int(k, 2) for k in result.keys()])
        probabilities = counts / self.shots
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

def QuantumFCL() -> QuantumFullyConnectedLayer:
    simulator = qiskit.Aer.get_backend("qasm_simulator")
    return QuantumFullyConnectedLayer(1, simulator, shots=1024)

# ------------------------------------------------------------------
# Main hybrid quantum class
# ------------------------------------------------------------------
class FraudDetectionHybrid:
    """
    Quantum‑centric hybrid model that mirrors the classical structure.
    Each block is a variational circuit or a parameterised gate set.
    """
    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_params: Dict[str, np.ndarray],
        classifier_params: Dict[str, int],
    ) -> None:
        self.program = build_fraud_detection_program(fraud_params, fraud_layers)
        self.attention = QuantumSelfAttention(attention_params["n_qubits"])
        self.classifier, self.enc, self.wsize, self.obs = build_classifier_circuit(
            classifier_params["num_qubits"], classifier_params["depth"]
        )
        self.fcl = QuantumFCL()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.attention_params = attention_params

    def run_fraud(self, inputs: Sequence[float]) -> np.ndarray:
        """Execute the photonic fraud‑detection ansatz."""
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(self.program, data=inputs)
        return result.state

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> Dict[str, int]:
        """Execute the self‑attention quantum circuit."""
        return self.attention.run(self.backend, rotation_params, entangle_params, shots)

    def run_classifier(
        self,
        enc_params: Sequence[float],
        weight_params: Sequence[float],
    ) -> np.ndarray:
        """Bind all variational parameters and evaluate the classifier."""
        bound = {param: val for param, val in zip(self.enc, enc_params)}
        bound.update({param: val for param, val in zip(self.wsize, weight_params)})
        circuit = self.classifier.bind_parameters(bound)
        job = qiskit.execute(circuit, self.backend, shots=1024)
        result = job.result()
        # Simple expectation: average of Z counts weighted by +/-1
        expectation = 0.0
        for bitstring, count in result.get_counts(circuit).items():
            z = 1 if bitstring.count("1") % 2 == 0 else -1
            expectation += z * count
        expectation /= 1024
        return np.array([expectation])

    def run_fcl(self, thetas: Iterable[float]) -> np.ndarray:
        """Run the quantum fully‑connected sub‑circuit."""
        return self.fcl.run(thetas)

    def run_full(
        self,
        inputs: Sequence[float],
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, int], np.ndarray, np.ndarray]:
        """Execute the full hybrid pipeline."""
        fraud_out = self.run_fraud(inputs)
        attention_counts = self.run_attention(rotation_params, entangle_params)
        cls_out = self.run_classifier(rotation_params, entangle_params)
        fcl_out = self.run_fcl([0.5])
        return fraud_out, attention_counts, cls_out, fcl_out

__all__ = [
    "FraudDetectionHybrid",
    "build_fraud_detection_program",
    "QuantumSelfAttention",
    "build_classifier_circuit",
    "QuantumFCL",
]
