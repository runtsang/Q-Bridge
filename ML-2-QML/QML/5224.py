"""
Hybrid quantum‑classical classifier – quantum implementation.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Optional
import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, transpile, assemble, Aer
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import torchquantum as tq
from torchquantum.functional import func_name_dict

# ----------------------------------------------------------------------
# Quantum circuit builder (incremental data‑uploading)
# ----------------------------------------------------------------------
def build_classifier_circuit(num_qubits: int,
                             depth: int = 2,
                             encoding_type: str = "rx") -> tuple[QuantumCircuit, list[ParameterVector], list[ParameterVector], list[SparsePauliOp]]:
    """
    Construct a layered ansatz with an explicit encoding and variational parameters.
    Returns (circuit, encoding, weights, observables).
    """
    if encoding_type == "rx":
        encoding = ParameterVector("x", num_qubits)
    else:
        raise ValueError("Unsupported encoding type")

    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]

    return circuit, list(encoding), list(weights), observables


# ----------------------------------------------------------------------
# Quantum kernel using TorchQuantum
# ----------------------------------------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a programmable list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice,
                x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
    """Evaluate the Gram matrix between datasets ``a`` and ``b``."""
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ----------------------------------------------------------------------
# Quantum transformer block (TorchQuantum)
# ----------------------------------------------------------------------
class TransformerBlockQuantum(tq.QuantumModule):
    """A minimal transformer block realised entirely in TorchQuantum."""
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int,
                 n_qubits: int = 8, n_qlayers: int = 1):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        # Simple feed‑forward via a quantum circuit
        self.ffn = tq.QuantumModule()
        self.ffn.register_module("encoder", tq.GeneralEncoder(
            [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_qubits)]
        ))
        self.ffn.register_module("params",
                                 tq.ModuleList([tq.RY(has_params=True, trainable=True)
                                                for _ in range(n_qubits)]))
        self.ffn.register_module("measure", tq.MeasureAll(tq.PauliZ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x shape (batch, embed_dim).  We map each token to a qubit.
        batch = x.shape[0]
        qdev = self.q_device.copy(bsz=batch)
        self.ffn.encoder(qdev, x)
        for gate in self.ffn.params:
            gate(qdev)
        return self.ffn.measure(qdev)


# ----------------------------------------------------------------------
# Hybrid quantum classifier
# ----------------------------------------------------------------------
class HybridQuantumClassifier:
    """
    Quantum implementation of the hybrid classifier.  It can operate either
    purely as a parameterised quantum circuit or embed a quantum transformer
    block.  The interface mirrors the classical version so that a single
    training script can switch implementations.
    """
    def __init__(self,
                 num_qubits: int,
                 depth: int = 2,
                 use_transformer: bool = False,
                 transformer_params: Optional[dict] = None,
                 use_kernel: bool = False):
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_transformer = use_transformer
        self.use_kernel = use_kernel

        if use_transformer:
            params = transformer_params or {}
            embed_dim = params.get("embed_dim", num_qubits)
            num_heads = params.get("num_heads", 4)
            ffn_dim = params.get("ffn_dim", 4 * embed_dim)
            self.transformer = TransformerBlockQuantum(embed_dim,
                                                       num_heads,
                                                       ffn_dim,
                                                       n_qubits=num_qubits,
                                                       n_qlayers=1)
            self.classifier = tq.QuantumModule()
            self.classifier.register_module("encoder",
                                            tq.GeneralEncoder(
                                                [{"input_idx": [i], "func": "ry", "wires": [i]}
                                                 for i in range(num_qubits)]))
            self.classifier.register_module("measure", tq.MeasureAll(tq.PauliZ))
        else:
            self.circuit, self.encoding, self.weights, self.observables = build_classifier_circuit(
                num_qubits, depth)
            self.backend = Aer.get_backend("qasm_simulator")
            self.shots = 1024

        if use_kernel:
            self.kernel = Kernel()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the quantum circuit and return expectation values."""
        if self.use_transformer:
            # In a real setting we would feed data through the transformer.
            # Here we simply run the classifier circuit.
            qdev = self.classifier.q_device.copy(bsz=1)
            self.classifier.encoder(qdev, torch.tensor(thetas))
            return self.classifier.measure(qdev).numpy()
        else:
            params = {self.encoding[i]: thetas[i] for i in range(len(thetas))}
            bound = self.circuit.assign_parameters(params)
            transpiled = transpile(bound, self.backend)
            qobj = assemble(transpiled, shots=self.shots)
            result = self.backend.run(qobj).result()
            counts = result.get_counts()
            probs = np.array(list(counts.values())) / self.shots
            expectation = np.sum(np.array(list(counts.keys()), dtype=float) * probs)
            return np.array([expectation])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        if not self.use_kernel:
            raise RuntimeError("Kernel functionality not enabled.")
        return kernel_matrix(a, b)

__all__ = [
    "build_classifier_circuit",
    "HybridQuantumClassifier",
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
]
