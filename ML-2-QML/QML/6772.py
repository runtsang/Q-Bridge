"""Quantum self‑attention module that fuses a variational kernel ansatz with a Qiskit‑based
attention circuit.  It extends the original SelfAttention quantum seed by embedding
a TorchQuantum kernel within the attention circuit so that the attention weights are
derived from a quantum kernel evaluation.  The class follows the same interface as
the classical version to ease comparison.

The implementation uses Qiskit Aer for simulation and TorchQuantum for the
kernel ansatz.  The kernel is evaluated on a quantum device and the resulting
overlap is used as the attention similarity matrix.  The circuit then applies
parameter‑dependent rotations and controlled‑X gates to encode the query,
key, and value states before measurement.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict

class Kernel(tq.QuantumModule):
    """TorchQuantum kernel ansatz identical to the QML seed."""
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        # Simple rotation‑only ansatz per wire
        self.ansatz = tq.QuantumModule()
        self.ansatz.add_layer(tq.RY, wires=list(range(self.n_wires)))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        # Encode x
        for i in range(self.n_wires):
            tq.RY(q_device, wires=[i], params=x[:, i])
        # Encode y with negative parameters
        for i in range(self.n_wires):
            tq.RY(q_device, wires=[i], params=-y[:, i])
        # Overlap will be extracted from q_device.states

class SelfAttentionKernel:
    """Quantum self‑attention using a variational kernel ansatz."""
    def __init__(self, n_qubits: int = 4, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        # Instantiate the quantum kernel
        self.kernel = Kernel(n_wires=n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray,
                       entangle_params: np.ndarray,
                       input_row: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        # Encode input via rotation gates
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        # Entangle qubits
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        # Optional: encode the actual input data as additional rotations
        for i in range(self.n_qubits):
            circuit.ry(input_row[0, i], i)
        circuit.measure(self.qr, self.cr)
        return circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        """
        Execute the quantum self‑attention circuit and return a weighted
        probability distribution over the output states for each input sample.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the query rotation gates.
        entangle_params : np.ndarray
            Parameters for the entangling gates.
        inputs : np.ndarray
            Input data of shape (batch, n_qubits).

        Returns
        -------
        np.ndarray
            Weighted probability vectors of shape (batch, 2**n_qubits).
        """
        batch = inputs.shape[0]
        # Compute kernel similarity matrix between all pairs of inputs
        sim = np.zeros((batch, batch))
        for i in range(batch):
            for j in range(batch):
                sim[i, j] = self.kernel(inputs[i], inputs[j]).item()
        # Attention weights via softmax
        attn = np.exp(sim / np.sqrt(self.n_qubits))
        attn = attn / attn.sum(axis=1, keepdims=True)

        # Compute measurement probabilities for each sample
        probs_list = []
        for i in range(batch):
            circ = self._build_circuit(rotation_params, entangle_params, inputs[i:i+1])
            job = qiskit.execute(circ, self.backend, shots=self.shots)
            counts = job.result().get_counts(circ)
            probs = np.zeros(2 ** self.n_qubits)
            for bitstring, count in counts.items():
                idx = int(bitstring[::-1], 2)  # Qiskit bitstring order
                probs[idx] = count / self.shots
            probs_list.append(probs)
        probs_arr = np.stack(probs_list)  # shape (batch, 2**n_qubits)

        # Weighted sum across samples
        output = attn @ probs_arr
        return output

def SelfAttention():
    """Factory returning a quantum self‑attention instance."""
    return SelfAttentionKernel(n_qubits=4)
