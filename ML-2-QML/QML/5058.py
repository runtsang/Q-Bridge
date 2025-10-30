import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator
from typing import Tuple

# ---------- Autoencoder utilities (copy of ML version) ----------
class AutoencoderConfig:
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        enc_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                enc_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)
        dec_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                dec_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

# ---------- Quantum quanvolution ----------
class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Applies a random two‑qubit quantum kernel to 2×2 image patches.
    """
    def __init__(self, n_wires: int = 4):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.random_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
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
                self.random_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

# ---------- Quantum regression head ----------
class QuantumRegressionHead(tq.QuantumModule):
    """
    Uses a single‑qubit EstimatorQNN to map quantum features to a scalar.
    """
    def __init__(self, num_outputs: int = 1):
        super().__init__()
        self.num_outputs = num_outputs
        # Build a simple EstimatorQNN for each output (here one)
        self.estimators = nn.ModuleList()
        self.estimators.append(
            EstimatorQNN(
                circuit=self._build_circuit(),
                observables=[SparsePauliOp.from_list([("Z", 1)])],
                input_params=[Parameter("x")],
                weight_params=[Parameter("w")],
                estimator=StatevectorEstimator()
            )
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(Parameter("x"), 0)
        qc.rx(Parameter("w"), 0)
        return qc

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        preds = []
        for i in range(self.num_outputs):
            inp = { "x": features[:, i] }
            pred = self.estimators[0](inp)
            preds.append(pred)
        return torch.stack(preds, dim=1)

# ---------- Hybrid quantum model ----------
class QuanvolutionHybrid(tq.QuantumModule):
    """
    Quantum‑augmented version of the classical hybrid model.
    Replaces the 2‑D convolution with a quanvolution filter and the
    regression head with a quantum EstimatorQNN.  An autoencoder still
    compresses the feature vector before regression.
    """
    def __init__(
        self,
        n_wires: int = 4,
        latent_dim: int = 32,
        autoencoder_hidden: Tuple[int, int] = (128, 64),
        dropout: float = 0.1,
        regression_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.quantum_conv = QuantumQuanvolutionFilter(n_wires)
        dummy = torch.zeros(1, 1, 28, 28)
        conv_out = self.quantum_conv(dummy).view(1, -1)
        self.conv_output_dim = conv_out.shape[1]
        ae_cfg = AutoencoderConfig(
            input_dim=self.conv_output_dim,
            latent_dim=latent_dim,
            hidden_dims=autoencoder_hidden,
            dropout=dropout,
        )
        self.autoencoder = AutoencoderNet(ae_cfg)
        self.quantum_head = QuantumRegressionHead(num_outputs=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.quantum_conv(x)
        flat = conv_out.view(conv_out.size(0), -1)
        compressed = self.autoencoder.encode(flat)
        out = self.quantum_head(compressed)
        return F.log_softmax(out, dim=-1) if out.shape[-1] > 1 else out

__all__ = ["QuanvolutionHybrid"]
