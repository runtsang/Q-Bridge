import torch
from torch import nn
import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit

# --------------------------------------------------------------------------- #
# Classical convolution filter (drop‑in replacement for quanvolution)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # data is a 2‑D tensor of shape (H, W)
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# --------------------------------------------------------------------------- #
# Quantum convolution filter
# --------------------------------------------------------------------------- #
class QuanvCircuit:
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float) -> None:
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0 for i, val in enumerate(dat)}
            param_binds.append(bind)
        job = qiskit.execute(self._circuit, self.backend, shots=self.shots, parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)
        counts = sum(sum(int(bit) for bit in key) * val for key, val in result.items())
        return counts / (self.shots * self.n_qubits)

# --------------------------------------------------------------------------- #
# Classical self‑attention helper
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention:
    def __init__(self, embed_dim: int = 4):
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# --------------------------------------------------------------------------- #
# Fraud‑detection style feed‑forward block
# --------------------------------------------------------------------------- #
class FraudLayerParameters:
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias   = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias   = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters]
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Unified hybrid component
# --------------------------------------------------------------------------- #
class ConvHybrid(nn.Module):
    """
    A drop‑in replacement that can operate in either classical or quantum mode,
    optionally augments the feature map with self‑attention, and finally passes
    the result through a fraud‑detection style feed‑forward network if parameters
    are supplied.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_attention: bool = True,
        fraud_params: list[FraudLayerParameters] | None = None,
        mode: str = "classical"  # or "quantum"
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "classical":
            self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
        else:
            self.conv = QuanvCircuit(kernel_size, qiskit.Aer.get_backend("qasm_simulator"),
                                     shots=100, threshold=threshold)
        self.use_attention = use_attention
        if use_attention:
            self.attn = ClassicalSelfAttention(embed_dim=kernel_size)
        self.fraud_net = build_fraud_detection_program(fraud_params[0], fraud_params[1:]) if fraud_params else None

    def forward(self, x: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            2‑D array (H, W) or a batch of such arrays.

        Returns
        -------
        torch.Tensor
            Feature vector(s) after convolution, optional attention, and fraud‑detection head.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        outputs = []
        for sample in x:
            data = sample.squeeze().numpy()
            if self.mode == "classical":
                feat = self.conv(data)
            else:
                feat = self.conv.run(data)
            if self.use_attention:
                feat = self.attn.run(np.array([0]), np.array([0]), np.array([feat]))
                feat = feat[0]
            if self.fraud_net:
                feat = self.fraud_net(torch.tensor(feat, dtype=torch.float32))
            outputs.append(feat)
        return torch.stack(outputs)

__all__ = ["ConvHybrid", "FraudLayerParameters", "build_fraud_detection_program"]
