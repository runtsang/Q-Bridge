import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import Aer, execute, QuantumCircuit, Parameter
from qiskit.circuit.random import random_circuit
import numpy as np

class QuanvCircuit:
    """
    Quantum convolution (quanvolution) filter that measures average |1> probability
    for a 2×2 patch encoded via RX rotations.
    """
    def __init__(self, kernel_size: int = 2, shots: int = 100, threshold: float = 127):
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = Aer.get_backend('qasm_simulator')
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, depth=2)
        self._circuit.measure_all()

    def run(self, patch: np.ndarray) -> np.ndarray:
        """
        Run the circuit on a single 2×2 patch.
        Returns a probability vector of length n_qubits.
        """
        patch = patch.reshape(1, self.n_qubits)
        param_binds = []
        for dat in patch:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(self._circuit, self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        probs = np.zeros(self.n_qubits)
        for bitstring, count in result.items():
            for i, bit in enumerate(reversed(bitstring)):
                if bit == '1':
                    probs[i] += count
        probs /= self.shots
        return probs

class SimpleEncoder(tq.QuantumModule):
    """
    Encodes a classical vector into qubit states via RX rotations.
    """
    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.rx = tq.RX(has_params=True, trainable=False)

    def forward(self, qdev: tq.QuantumDevice, data: torch.Tensor):
        # data shape: (bsz, n_qubits)
        for i in range(self.n_qubits):
            self.rx(qdev, wires=i, params=data[..., i])

class QFCQuantumLayer(tq.QuantumModule):
    """
    Quantum fully‑connected layer inspired by the Quantum‑NAT QLayer.
    """
    def __init__(self, output_dim: int = 4):
        super().__init__()
        self.n_qubits = output_dim
        self.random_layer = tq.RandomLayer(n_ops=30, wires=range(self.n_qubits))
        self.rx = tq.RX(has_params=True, trainable=True)
        self.ry = tq.RY(has_params=True, trainable=True)
        self.rz = tq.RZ(has_params=True, trainable=True)
        self.crx = tq.CRX(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
        self.random_layer(qdev)
        self.rx(qdev, wires=0)
        self.ry(qdev, wires=1)
        self.rz(qdev, wires=2)
        self.crx(qdev, wires=[0, 3])
        return self.measure(qdev)

class ConvGen221(tq.QuantumModule):
    """
    Quantum implementation of ConvGen221.
    Applies a quanvolution filter to image patches and then a quantum
    fully‑connected layer for classification.
    """
    def __init__(self, output_dim: int = 4, shots: int = 200):
        super().__init__()
        self.quanv = QuanvCircuit(kernel_size=2, shots=shots, threshold=127)
        self.encoder = SimpleEncoder(n_qubits=4)
        self.qfc = QFCQuantumLayer(output_dim=output_dim)
        self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (bsz, 1, H, W)
        bsz, _, H, W = x.shape
        # Extract 2×2 patches with stride 2
        patches = torch.nn.functional.unfold(
            x, kernel_size=2, stride=2, padding=0
        )  # shape: (bsz, 4, L)
        patches = patches.permute(0, 2, 1).reshape(-1, 4)  # (bsz*L, 4)
        probs_list = []
        for patch in patches.cpu().numpy():
            probs_list.append(self.quanv.run(patch))
        probs = torch.tensor(np.stack(probs_list), dtype=x.dtype, device=x.device)  # (bsz*L, 4)

        # Quantum fully‑connected layer
        qdev = tq.QuantumDevice(n_wires=4, bsz=probs.shape[0],
                                device=x.device, record_op=False)
        self.encoder(qdev, probs)
        out = self.qfc(qdev)  # (bsz*L, 4)
        out = self.norm(out)
        # Aggregate spatially
        out = out.reshape(bsz, -1, 4).mean(dim=1)  # (bsz, 4)
        return out

__all__ = ["ConvGen221"]
