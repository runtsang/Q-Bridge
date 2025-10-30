import torch
import torchquantum as tq
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Sequence, Iterable

# --------------------------------------------------------------------
# Quantum kernel utilities (inspired by QuantumKernelMethod)
# --------------------------------------------------------------------
from torchquantum.functional import func_name_dict, op_name_dict

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data through a list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
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
    kernel = Kernel()
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# --------------------------------------------------------------------
# Patch encoder and estimator (combining Quanvolution + EstimatorQNN)
# --------------------------------------------------------------------
class QuantumPatchEncoder(tq.QuantumModule):
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

    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
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
                self.encoder(q_device, data)
                self.q_layer(q_device)
                measurement = self.measure(q_device)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class EstimatorQNN(tq.QuantumModule):
    """Small parametric quantum circuit used as a neural‑network head."""
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(self.n_wires)
        self.weight = nn.Parameter(torch.randn(1))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, q_device: tq.QuantumDevice, inputs: torch.Tensor) -> torch.Tensor:
        bsz = inputs.shape[0]
        q_device.reset_states(bsz)
        tq.apply(tq.RY, q_device, wires=[0], params=inputs.squeeze(-1))
        tq.apply(tq.RX, q_device, wires=[1], params=self.weight.expand(bsz))
        measurement = self.measure(q_device)
        return measurement[:, 1]  # expectation of the second qubit

class HybridQuanvolution(tq.QuantumModule):
    """Hybrid quantum‑classical network that fuses a quanvolution encoder with a quantum estimator."""
    def __init__(self):
        super().__init__()
        self.encoder = QuantumPatchEncoder()
        self.estimator = EstimatorQNN()
        self.classifier = nn.Linear(1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qdev = tq.QuantumDevice(self.encoder.n_wires)
        features = self.encoder(qdev, x)          # (B, 4*14*14)
        inputs = features.mean(dim=1, keepdim=True)  # (B,1)
        logits = self.estimator(qdev, inputs.squeeze(-1))  # (B,)
        logits = self.classifier(logits.unsqueeze(-1))     # (B,10)
        return F.log_softmax(logits, dim=-1)

# --------------------------------------------------------------------
# Evaluation primitives (FastBaseEstimator + FastEstimator)
# --------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: tq.QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        observables = list(observables)
        results: List[List[float]] = []
        for values in parameter_sets:
            circ = self._bind(values)
            state = tq.Statevector.from_instruction(circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class FastEstimator(FastBaseEstimator):
    """Adds Gaussian shot noise to the deterministic estimator."""
    def evaluate(self, observables: Iterable[tq.QuantumOperator], parameter_sets: Sequence[Sequence[float]],
                 shots: int | None = None, seed: int | None = None) -> List[List[float]]:
        raw = super().evaluate(observables, parameter_sets)
        if shots is None:
            return raw
        rng = np.random.default_rng(seed)
        noisy = []
        for row in raw:
            noisy_row = [float(rng.normal(mean, max(1e-6, 1 / shots))) for mean in row]
            noisy.append(noisy_row)
        return noisy

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "QuantumPatchEncoder",
    "EstimatorQNN",
    "HybridQuanvolution",
    "FastEstimator",
]
