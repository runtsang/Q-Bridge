import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, execute
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RealAmplitudes
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.quantum_info import Statevector
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

# --------------------------------------------------------------------------- #
# Quantum self‑attention circuit
# --------------------------------------------------------------------------- #
class QuantumSelfAttention:
    """Quantum circuit that mimics a self‑attention block."""

    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circ = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            circ.rx(rotation_params[3 * i], i)
            circ.ry(rotation_params[3 * i + 1], i)
            circ.rz(rotation_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)
        circ.measure(self.qr, self.cr)
        return circ

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024):
        circ = self._build(rotation_params, entangle_params)
        job = execute(circ, self.backend, shots=shots)
        return job.result().get_counts(circ)


# --------------------------------------------------------------------------- #
# Quantum auto‑encoder (SamplerQNN)
# --------------------------------------------------------------------------- #
class QuantumAutoEncoder:
    """Quantum auto‑encoder implemented with a SamplerQNN."""

    def __init__(self, num_latent: int = 3, num_trash: int = 2):
        algorithm_globals.random_seed = 42
        self.sampler = qiskit.primitives.StatevectorSampler()
        def ansatz(num_qubits):
            return RealAmplitudes(num_qubits, reps=5)

        def auto_encoder_circuit(num_latent, num_trash):
            qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = ClassicalRegister(1, "c")
            qc = QuantumCircuit(qr, cr)
            qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
            qc.barrier()
            aux = num_latent + 2 * num_trash
            qc.h(aux)
            for i in range(num_trash):
                qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
            qc.h(aux)
            qc.measure(aux, cr[0])
            return qc

        self.circuit = auto_encoder_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def run(self, weight_values: np.ndarray) -> np.ndarray:
        """Execute the auto‑encoder with the supplied weight values."""
        return self.qnn.forward(weight_values)


# --------------------------------------------------------------------------- #
# Quantum fully‑connected model (QFCModel)
# --------------------------------------------------------------------------- #
class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected model inspired by the Quantum‑NAT paper."""

    class QLayer(tq.QuantumModule):
        def __init__(self):
            super().__init__()
            self.n_wires = 4
            self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(self.n_wires)))
            self.rx0 = tq.RX(has_params=True, trainable=True)
            self.ry0 = tq.RY(has_params=True, trainable=True)
            self.rz0 = tq.RZ(has_params=True, trainable=True)
            self.crx0 = tq.CRX(has_params=True, trainable=True)

        @tq.static_support
        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            self.rx0(qdev, wires=0)
            self.ry0(qdev, wires=1)
            self.rz0(qdev, wires=3)
            self.crx0(qdev, wires=[0, 2])
            tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
            tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
            tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)

    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = self.QLayer()
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(self.n_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
        pooled = F.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)


# --------------------------------------------------------------------------- #
# Quantum quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilterQuantum(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""

    def __init__(self):
        super().__init__()
        self.n_wires = 4
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


# --------------------------------------------------------------------------- #
# Hybrid pipeline (quantum branch)
# --------------------------------------------------------------------------- #
class SelfAttentionGen074:
    """
    Quantum branch of the hybrid pipeline.  The interface matches the
    classical version to enable side‑by‑side experimentation.  It
    contains a quantum self‑attention circuit, a quantum auto‑encoder,
    a QFCModel, and a quantum quanvolution filter.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 num_latent: int = 3,
                 num_trash: int = 2,
                 num_classes: int = 10):
        self.quantum_attention = QuantumSelfAttention(n_qubits)
        self.quantum_autoencoder = QuantumAutoEncoder(num_latent, num_trash)
        self.qfc_model = QFCModel()
        self.quantum_filter = QuanvolutionFilterQuantum()
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

    # ---------- Quantum path --------------------------------------------------
    def run_quantum(self,
                    rotation_params: np.ndarray,
                    entangle_params: np.ndarray,
                    weight_values: np.ndarray,
                    input_tensor: torch.Tensor,
                    shots: int = 1024) -> torch.Tensor:
        """
        Run a full quantum forward pass.

        Parameters
        ----------
        rotation_params : np.ndarray
            Parameters for the attention circuit.
        entangle_params : np.ndarray
            Entanglement parameters for the attention circuit.
        weight_values : np.ndarray
            Weight values for the quantum auto‑encoder.
        input_tensor : torch.Tensor
            4‑D image tensor (B, 1, 28, 28) or 3‑D sequence tensor.
        shots : int, optional
            Number of shots for the attention circuit.
        Returns
        -------
        torch.Tensor
            Log‑probabilities from the classifier.
        """
        # Attention output (classical counts -> tensor)
        attn_counts = self.quantum_attention.run(rotation_params, entangle_params, shots)
        # Convert counts to a deterministic tensor (e.g. total probability)
        attn_tensor = torch.tensor(
            [sum(v for k, v in attn_counts.items()) / shots], dtype=torch.float32
        ).unsqueeze(0)

        # Auto‑encoder output
        ae_output = self.quantum_autoencoder.run(weight_values)
        ae_tensor = torch.tensor(ae_output, dtype=torch.float32)

        # Combine with quantum filter
        qfilter_out = self.quantum_filter(input_tensor)

        # Linear classifier
        logits = self.classifier(torch.cat([attn_tensor, qfilter_out], dim=1))
        return F.log_softmax(logits, dim=-1)

    # ---------- Classical path placeholder ------------------------------------
    def run_classical(self, *args, **kwargs):
        raise NotImplementedError("Classical execution is implemented in the classical module.")


__all__ = ["SelfAttentionGen074"]
