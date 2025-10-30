import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class HybridQuantumNat(tq.QuantumModule):
    """Quantum hybrid model that mirrors the classical HybridQuantumNat architecture."""

    class QEncoder(tq.QuantumModule):
        """Quantum encoder inspired by QFCModel."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
            self.q_layer = self.QLayer()
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.norm = nn.BatchNorm1d(n_wires)

        class QLayer(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device, record_op=True)
            pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
            self.encoder(qdev, pooled)
            self.q_layer(qdev)
            out = self.measure(qdev)
            return self.norm(out)

    class QSampler(tq.QuantumModule):
        """Qiskit-based sampler network."""
        def __init__(self, n_qubits: int):
            super().__init__()
            self.n_qubits = n_qubits
            inputs = ParameterVector("input", 2)
            weights = ParameterVector("weight", 4)
            qc = QuantumCircuit(2)
            qc.ry(inputs[0], 0)
            qc.ry(inputs[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[0], 0)
            qc.ry(weights[1], 1)
            qc.cx(0, 1)
            qc.ry(weights[2], 0)
            qc.ry(weights[3], 1)
            self.sampler = Sampler()
            self.qnn = SamplerQNN(circuit=qc,
                                  input_params=inputs,
                                  weight_params=weights,
                                  sampler=self.sampler)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.qnn.forward(x)

    class QCNNLayer(tq.QuantumModule):
        """Simple quantum convolutional layer using a random circuit."""
        def __init__(self, n_qubits: int, depth: int = 3):
            super().__init__()
            self.n_qubits = n_qubits
            self.layers = nn.ModuleList([tq.RandomLayer(n_ops=20, wires=list(range(n_qubits)))
                                         for _ in range(depth)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, qdev: tq.QuantumDevice) -> torch.Tensor:
            for layer in self.layers:
                layer(qdev)
            return self.measure(qdev)

    class QLSTMCell(tq.QuantumModule):
        """Quantum LSTM cell where gates are implemented by small circuits."""
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [{"input_idx": [0], "func": "rx", "wires": [0]},
                 {"input_idx": [1], "func": "rx", "wires": [1]},
                 {"input_idx": [2], "func": "rx", "wires": [2]},
                 {"input_idx": [3], "func": "rx", "wires": [3]}]
            )
            self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires):
                tgt = 0 if wire == self.n_wires - 1 else wire + 1
                tqf.cnot(qdev, wires=[wire, tgt])
            return self.measure(qdev)

    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.encoder = self.QEncoder(n_wires)
        self.sampler = self.QSampler(n_wires)
        self.cnn_layer = self.QCNNLayer(n_wires)
        self.lstm_cell = self.QLSTMCell(n_wires)
        self.tag_head = nn.Linear(n_wires, 10)

    def forward(self,
                images: torch.Tensor,
                seq: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass mirroring the classical HybridQuantumNat.

        Args:
            images: Tensor of shape (B, C, H, W).
            seq: Optional sequence tensor of shape (B, T,?) for tagging.

        Returns:
            If seq is None: Tensor of shape (B, n_wires) representing the LSTM cell output.
            If seq is provided: Tuple[Tensor, Tensor] where first element is LSTM output
            and second element is tag logits of shape (B, T, 10).
        """
        img_out = self.encoder(images)
        samp_out = self.sampler(img_out)
        # Encode sampler output into a quantum device for the QCNN layer
        qdev = tq.QuantumDevice(n_wires=self.lstm_cell.n_wires,
                                bsz=samp_out.shape[0],
                                device=samp_out.device)
        self.encoder.encoder(qdev, samp_out)
        qcnn_out = self.cnn_layer(qdev)
        lstm_out = self.lstm_cell(qcnn_out)

        if seq is not None:
            tag_logits = self.tag_head(lstm_out).unsqueeze(1).repeat(1, seq.size(1), 1)
            return lstm_out, tag_logits
        return lstm_out
