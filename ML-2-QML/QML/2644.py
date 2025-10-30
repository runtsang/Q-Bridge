import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
import torchquantum as tq
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridAttentionQuanvolution:
    """
    Quantum hybrid module that fuses a quanvolution filter (implemented with
    torchquantum) and a self‑attention style circuit (implemented with qiskit).
    The public API mirrors the classical version so that experiment scripts
    can be swapped without modification.
    """
    def __init__(self, n_qubits: int = 4, n_wires: int = 4):
        self.n_qubits = n_qubits
        self.n_wires = n_wires
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")
        # Quanvolution components
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(self.n_wires * 14 * 14, 10)

    def _attention_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)
        circuit.measure(self.qr, self.cr)
        return circuit

    def _quanvolution(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply a random two‑qubit quantum kernel to 2×2 image patches.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

    def run(self, backend: qiskit.providers.baseprovider.BaseBackend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024,
            inputs: torch.Tensor = None) -> np.ndarray:
        """
        Execute the combined quanvolution + attention circuit.
        Parameters
        ----------
        backend : qiskit backend
            Quantum backend to execute the attention circuit.
        rotation_params, entangle_params : np.ndarray
            Parameters for the attention circuit.
        shots : int
            Number of shots for the attention circuit.
        inputs : torch.Tensor
            Input images of shape (B,1,28,28).  If None, a dummy tensor
            of zeros is used.
        Returns
        -------
        logits : np.ndarray
            Log‑probabilities over 10 classes.
        """
        if inputs is None:
            inputs = torch.zeros((1, 1, 28, 28))
        # Quanvolution feature extraction
        features = self._quanvolution(inputs)
        # Attention circuit
        attn_circ = self._attention_circuit(rotation_params, entangle_params)
        job = execute(attn_circ, backend, shots=shots)
        counts = job.result().get_counts(attn_circ)
        # Convert counts to a probability vector of size 2**n_qubits
        probs = np.zeros(2 ** self.n_qubits)
        for bitstring, c in counts.items():
            idx = int(bitstring[::-1], 2)
            probs[idx] = c / shots
        # Combine with quanvolution features (simple concatenation)
        combined = torch.cat([torch.tensor(probs, dtype=torch.float32).unsqueeze(0), features], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1).detach().cpu().numpy()

__all__ = ["HybridAttentionQuanvolution"]
