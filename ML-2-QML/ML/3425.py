from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# 1. Shared parameter container
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Unified description of a single layer used by both classical and photonic
    implementations.  The fields correspond one‑to‑one with the photonic
    gates used in the quantum circuit so that a single configuration can drive
    both branches."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. Classical sub‑network
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """
    Convert a FraudLayerParameters instance into a compact nn.Module.
    The first (input) layer is left unclipped to keep the expressive range
    of the photonic prototype; subsequent layers are clipped to the physical
    bounds of the quantum circuit.
    """
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)

    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    # The activation mimics the photonic amplitude‑phase mixing.
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.activation(self.linear(inputs))
            return x * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Assemble a multi‑stage neural network that mimics the photonic architecture.
    The network ends with a single‑output linear layer that produces a
    fraud‑risk score.
    """
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 3. Graph‑based quantum side
# --------------------------------------------------------------------------- #
def _tensored_id(num_qubits: int) -> torch.Tensor:
    """Return a 2^n × 2^n identity tensor in PyTorch for fast simulation."""
    dim = 2 ** num_qubits
    return torch.eye(dim, dtype=torch.complex64)

def _tensored_zero(num_qubits: int) -> torch.Tensor:
    """Return a zero‑state projector for a single qubit."""
    return torch.eye(1, 1, dtype=torch.complex64)

def _random_qubit_unitary(num_qubits: int) -> torch.Tensor:
    """Generate a Haar‑distributed random unitary using SciPy."""
    dim = 2 ** num_qubits
    mat = sc.random.normal(size=(dim, dim)) + 1j * sc.random.normal(size=(dim, dim))
    return torch.from_numpy(sc.linalg.orth(mat)).to(torch.complex64)

def _random_qubit_state(num_qubits: int) -> torch.Tensor:
    """Create a random pure state vector."""
    dim = 2 ** num_qubits
    vec = sc.random.normal(size=(dim, 1)) + 1j * sc.random.normal(size=(dim, 1))
    vec /= sc.linalg.norm(vec)
    return torch.from_numpy(vec).squeeze()

def random_training_data(unitary: torch.Tensor, samples: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate a dataset of input–output pairs for a given target unitary."""
    return [( _random_qubit_state(unitary.shape[0]).unsqueeze(0),
              unitary @ _random_qubit_state(unitary.shape[0]).unsqueeze(0))
             for _ in range(samples)]

def random_network(qnn_arch: list[int], samples: int):
    """Build a random layer‑wise unitary graph suitable for a QNN."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[torch.Tensor]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[torch.Tensor] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                # Append an extra identity to keep the correct tensor shape.
                op = torch.kron(op, _tensored_id(num_outputs - 1))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _swap_registers(tensor: torch.Tensor, source: int, target: int) -> torch.Tensor:
    """Reorder qubits in a tensor representation of a state or operator."""
    if source == target:
        return tensor
    order = list(range(tensor.shape[0]))
    order[source], order[target] = order[target], order[source]
    return tensor.permute(order)

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[torch.Tensor]],
    layer: int,
    input_state: torch.Tensor,
) -> torch.Tensor:
    """Apply a single QNN layer to a state and trace out the inputs."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]

    # Prepare the full input state for the layer.
    state = torch.kron(input_state, _tensored_zero(num_outputs))

    # Compose the overall unitary for the layer.
    layer_unitary = unitaries[layer][0].clone()
    for gate in unitaries[layer][1:]:
        layer_unitary = torch.kron(gate, layer_unitary)

    # Apply and trace out the inputs.
    full_state = layer_unitary @ state
    return _partial_trace_remove(full_state, range(num_inputs))

def _partial_trace_remove(state: torch.Tensor, remove: Sequence[int]) -> torch.Tensor:
    """Keep only the remaining qubits after partial trace."""
    keep = list(range(state.shape[0] - len(remove)))
    for idx in sorted(remove, reverse=True):
        keep.pop(idx)
    return state.reshape(*keep)

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[torch.Tensor]],
    samples: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> list[list[torch.Tensor]]:
    """Propagate a sample through the QNN and collect the state at each layer."""
    outputs = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for layer in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, layer, current)
            states.append(current)
        outputs.append(states)
    return outputs

def state_fidelity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return the squared overlap between two pure states."""
    a_norm = a / (torch.norm(a) + 1e-12)
    b_norm = b / (torch.norm(b) + 1e-12)
    return float((a_norm.conj().t() @ b_norm).item().real ** 2)

def fidelity_adjacency(
    states: Sequence[torch.Tensor],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> torch.Tensor:
    """Construct a weighted adjacency matrix from state fidelities."""
    n = len(states)
    mat = torch.zeros((n, n), dtype=torch.float32)
    for i in range(n):
        for j in range(i + 1, n):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                mat[i, j] = mat[j, i] = 1.0
            elif secondary is not None and fid >= secondary:
                mat[i, j] = mat[j, i] = secondary_weight
    return mat

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "random_network",
    "random_training_data",
    "feedforward",
    "state_fidelity",
    "fidelity_adjacency",
]
