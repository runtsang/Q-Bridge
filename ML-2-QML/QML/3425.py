import pennylane as qml
from pennylane import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# 1. Shared dataclass for quantum circuit parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a photonic‑style quantum layer in PennyLane."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
# 2. PennyLane photonic‐style circuit
# --------------------------------------------------------------------------- #
def _apply_layer(q: Sequence[qml.Device], params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a single photonic layer to the device."""
    # Beam splitter
    qml.BSgate(params.bs_theta, params.bs_phi)(wires=[0, 1])
    # Phase shifters
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase)(wires=i)
    # Squeezing
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_use = r if not clip else _clip(r, 5)
        qml.Sgate(r_use, phi)(wires=i)
    # Second beam splitter
    qml.BSgate(params.bs_theta, params.bs_phi)(wires=[0, 1])
    # Phase shifters again
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase)(wires=i)
    # Displacement
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_use = r if not clip else _clip(r, 5)
        qml.Dgate(r_use, phi)(wires=i)
    # Kerr
    for i, k in enumerate(params.kerr):
        k_use = k if not clip else _clip(k, 1)
        qml.Kgate(k_use)(wires=i)

def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric interval."""
    return max(-bound, min(bound, value))

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dev: qml.Device,
) -> qml.QuantumNode:
    """Build a QNode that implements the layered photonic circuit."""
    def circuit():
        _apply_layer(dev, input_params, clip=False)
        for layer in layers:
            _apply_layer(dev, layer, clip=True)
        return qml.expval(qml.PauliZ(0))

    return qml.QNode(circuit, dev)

# --------------------------------------------------------------------------- #
# 3. Quantum‑graph utilities
# --------------------------------------------------------------------------- #
def _random_qubit_unitary(num_qubits: int) -> np.ndarray:
    """Generate a random unitary using NumPy."""
    dim = 2 ** num_qubits
    mat = np.random.randn(dim, dim) + 1j*np.random.randn(dim, dim)
    return np.linalg.qr(mat)[0]

def _random_qubit_state(num_qubits: int) -> np.ndarray:
    """Create a random pure state."""
    dim = 2 ** num_qubits
    vec = np.random.randn(dim) + 1j*np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec

def random_training_data(unitary: np.ndarray, samples: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate input‑output pairs for a target unitary."""
    return [( _random_qubit_state(unitary.shape[0]).reshape(-1,1),
              unitary @ _random_qubit_state(unitary.shape[0]).reshape(-1,1))
             for _ in range(samples)]

def random_network(qnn_arch: list[int], samples: int):
    """Build a random layer‑wise unitary graph for a QNN."""
    target_unitary = _random_qubit_unitary(qnn_arch[-1])
    training_data = random_training_data(target_unitary, samples)

    unitaries: list[list[np.ndarray]] = [[]]
    for layer in range(1, len(qnn_arch)):
        num_inputs = qnn_arch[layer - 1]
        num_outputs = qnn_arch[layer]
        layer_ops: list[np.ndarray] = []
        for output in range(num_outputs):
            op = _random_qubit_unitary(num_inputs + 1)
            if num_outputs > 1:
                op = np.kron(op, np.eye(2 ** (num_outputs - 1)))
                op = _swap_registers(op, num_inputs, num_inputs + output)
            layer_ops.append(op)
        unitaries.append(layer_ops)

    return qnn_arch, unitaries, training_data, target_unitary

def _swap_registers(mat: np.ndarray, source: int, target: int) -> np.ndarray:
    """Swap qubit indices in a matrix."""
    if source == target:
        return mat
    perm = np.arange(mat.shape[0])
    perm[source], perm[target] = perm[target], perm[source]
    return mat[perm][:, perm]

def _layer_channel(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    layer: int,
    input_state: np.ndarray,
) -> np.ndarray:
    """Apply a single QNN layer and trace out the inputs."""
    num_inputs = qnn_arch[layer - 1]
    num_outputs = qnn_arch[layer]
    state = np.kron(input_state, np.zeros((2 ** num_outputs, 1), dtype=complex))
    layer_unitary = unitaries[layer][0].copy()
    for gate in unitaries[layer][1:]:
        layer_unitary = gate @ layer_unitary
    full_state = layer_unitary @ state
    return _partial_trace_remove(full_state, num_inputs)

def _partial_trace_remove(state: np.ndarray, remove: int) -> np.ndarray:
    """Keep only the remaining qubits after partial trace."""
    return state[np.s_[remove:, :]]

def feedforward(
    qnn_arch: Sequence[int],
    unitaries: Sequence[Sequence[np.ndarray]],
    samples: Iterable[tuple[np.ndarray, np.ndarray]],
) -> list[list[np.ndarray]]:
    """Propagate a sample through the QNN and collect the state at each layer."""
    outputs = []
    for sample, _ in samples:
        states = [sample]
        current = sample
        for _ in range(1, len(qnn_arch)):
            current = _layer_channel(qnn_arch, unitaries, _, current)
            states.append(current)
        outputs.append(states)
    return outputs

def state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the squared overlap of two pure states."""
    return np.abs(a.conj().T @ b)[0,0] ** 2

def fidelity_adjacency(
    states: Sequence[np.ndarray],
    threshold: float,
    *,
    secondary: float | None = None,
    secondary_weight: float = 0.5,
) -> np.ndarray:
    """Return a weighted adjacency matrix from state fidelities."""
    n = len(states)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            fid = state_fidelity(states[i], states[j])
            if fid >= threshold:
                mat[i,j] = mat[j,i] = 1.0
            elif secondary is not None and fid >= secondary:
                mat[i,j] = mat[j,i] = secondary_weight
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
