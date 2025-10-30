import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, assemble, transpile
from qiskit.providers.aer import AerSimulator


def quantum_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute a quantum kernel between two sets of feature vectors.
    The implementation follows the TorchQuantum ansatz used in the
    original project: each feature is encoded into a ry gate
    and the second vector is encoded with a negative angle.
    """
    # Ensure tensors are 2‑D
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if y.ndim == 1:
        y = y.unsqueeze(0)

    n_wires = x.shape[-1]
    q_device = tq.QuantumDevice(n_wires=n_wires)

    def encode(vec: torch.Tensor):
        for i in range(n_wires):
            func_name_dict["ry"](q_device, wires=[i], params=vec[i])

    outputs = torch.zeros((x.size(0), y.size(0)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            q_device.reset_states(1)
            encode(xi)
            encode(-yj)
            outputs[i, j] = torch.abs(q_device.states.view(-1)[0])

    return outputs


class HybridCircuit:
    """
    A simple variational circuit used as the expectation head in the
    hybrid classifier. It mirrors the design from the reference
    ClassicalQuantumBinaryClassification module.
    """
    def __init__(self, n_qubits: int = 1, shots: int = 100):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = AerSimulator()
        self.circuit = QuantumCircuit(self.n_qubits)
        all_qubits = list(range(self.n_qubits))
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(all_qubits)
        self.circuit.barrier()
        self.circuit.ry(self.theta, all_qubits)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit for each angle in ``thetas`` and return
        the expectation value of the Z‑observable.
        """
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probs = counts / self.shots
            return np.sum(states * probs)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])


def quantum_attention(rotation_params: np.ndarray,
                     entangle_params: np.ndarray,
                     inputs: np.ndarray,
                     shots: int = 1024) -> np.ndarray:
    """
    Quantum self‑attention block implemented with Qiskit.
    The routine encodes the input data into a circuit of single‑qubit
    rotations followed by controlled‑X entangling gates.
    """
    n_qubits = inputs.shape[-1]
    qr = QuantumRegister(n_qubits, "q")
    cr = ClassicalRegister(n_qubits, "c")
    circuit = QuantumCircuit(qr, cr)

    # Rotation layer
    for i in range(n_qubits):
        circuit.rx(rotation_params[3 * i], i)
        circuit.ry(rotation_params[3 * i + 1], i)
        circuit.rz(rotation_params[3 * i + 2], i)

    # Entangling layer
    for i in range(n_qubits - 1):
        circuit.crx(entangle_params[i], i, i + 1)

    circuit.measure(qr, cr)

    backend = AerSimulator()
    compiled = transpile(circuit, backend)
    qobj = assemble(compiled, shots=shots)
    job = backend.run(qobj)
    result = job.result().get_counts(circuit)

    probs = np.zeros(n_qubits)
    for bitstring, count in result.items():
        idx = int(bitstring, 2)
        probs[idx] = count / shots
    return probs


__all__ = [
    "quantum_kernel",
    "HybridCircuit",
    "quantum_attention",
]
