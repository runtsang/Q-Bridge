"""FraudDetectionHybridModel: a hybrid ML/QML architecture for fraud detection.

The module exposes three public classes:
```
    * FraudDetectionHybridModel – classical backbone + optional Qiskit classifier.
    * HybridQuantumLSTM – quantum‑enhanced LSTM cell usable in any tagger.
    * FraudLayerParameters – dataclass used by both parts.
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Classical photonic‑style backbone
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clip values to keep linear weights bounded – matches the photonic clip."""
    return max(-bound, min(bound, value))

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool,
) -> nn.Module:
    """Create a single 2‑input/2‑output linear block that mimics the photonic layer."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 0.0)  # keep bias negative for consistency

    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(x))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_backbone(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Instantiate a sequential PyTorch model that follows the photonic structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))  # final output head
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 2. Quantum‑augmented measurement head
# --------------------------------------------------------------------------- #
def _build_quantum_classifier(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Build a Qiskit‑based linear‑parameterised circuit that keeps the quantum
    measurement statistics consistent with the feature set.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector
    from qiskit.quantum_info import SparsePauliOp

    encoding = ParameterVector("x", num_features)
    weights = ParameterVector("theta", num_features * depth)

    circuit = QuantumCircuit(num_features)
    for i, param in enumerate(encoding):
        circuit.rx(param, i)

    index = 0
    for _ in range(depth):
        for i in range(num_features):
            circuit.ry(weights[index], i)
            index += 1
        for i in range(num_features - 1):
            circuit.cz(i, i + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_features - i - 1))
        for i in range(num_features)
    ]

    return circuit, list(encoding), list(weights), observables

class FraudDetectionHybridModel(nn.Module):
    """
    Classic‑to‑quantum hybrid model that trains a normalised
    back‑bone and an *in‑place*‑tuned quantum circuit.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: List[FraudLayerParameters],
        q_depth: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = build_fraud_detection_backbone(input_params, layer_params)
        self.q_circuit, self.q_encoding, self.q_weights, self.q_obs = _build_quantum_classifier(
            num_features=2, depth=q_depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that first runs the classical backbone, then feeds the
        last two features into the quantum circuit and returns the weighted
        sum of the observables.
        """
        # Classical forward
        class_out = self.backbone(x)  # shape (batch, 1)

        # Prepare quantum input
        q_input = x[:, -2:]  # last two features are encoded
        # Convert to numpy for Qiskit simulation
        q_values = q_input.detach().cpu().numpy()

        # Bind parameters
        param_dict = {str(p): val for p, val in zip(self.q_encoding, q_values.T)}
        q_circ = self.q_circuit.bind_parameters(param_dict)

        # Simulate (using statevector for simplicity)
        from qiskit.quantum_info import Statevector
        state = Statevector.from_instruction(q_circ).data

        # Compute expectation values
        exp_vals = []
        for op in self.q_obs:
            exp_vals.append(op.eigenvalues().dot(state.conj() * state))

        # Combine classical and quantum outputs
        q_out = torch.tensor(exp_vals, dtype=torch.float32, device=x.device).sum()
        return class_out.squeeze() + q_out

# --------------------------------------------------------------------------- #
# 3. Reusable quantum‑enhanced LSTM cell
# --------------------------------------------------------------------------- #
class HybridQuantumLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM cell that mirrors the classical LSTM but replaces
    each gate with a small variational quantum circuit.
    """
    class _QLayer(nn.Module):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector

            # Simple encoding: rx for each input feature
            self.encoding = ParameterVector("enc", n_wires)
            self.circuit = QuantumCircuit(n_wires)
            for i in range(n_wires):
                self.circuit.rx(self.encoding[i], i)

            # Variational layer
            self.weights = ParameterVector("w", n_wires)
            for i in range(n_wires):
                self.circuit.ry(self.weights[i], i)

            # Entangling
            for i in range(n_wires - 1):
                self.circuit.cz(i, i + 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Bind parameters
            param_dict = {str(p): v for p, v in zip(self.encoding, x.T)}
            circ = self.circuit.bind_parameters(param_dict)
            from qiskit.quantum_info import Statevector
            sv = Statevector.from_instruction(circ).data
            return torch.tensor(sv.real, device=x.device)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self._QLayer(n_qubits)
        self.input = self._QLayer(n_qubits)
        self.update = self._QLayer(n_qubits)
        self.output = self._QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

__all__ = ["FraudDetectionHybridModel", "HybridQuantumLSTM", "FraudLayerParameters"]
