"""Hybrid estimator combining a TorchQuantum model and kernel utilities.

This module mirrors the classical implementation while operating on
quantum circuits.  It supports deterministic evaluation of
expectation values, optional shot noise, and a quantum‑or‑classical
kernel matrix.
"""

import numpy as np
import torch
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict
from torch.quantum_info import Statevector
from typing import Iterable, Sequence, Callable, List, Optional

# --------------------------------------------------------------------------- #
# 1. Quantum kernel utilities
# --------------------------------------------------------------------------- #

class KernalAnsatz(tq.QuantumModule):
    """Encodes classical data via a list of quantum gates."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class Kernel(tq.QuantumModule):
    """Fixed quantum kernel based on a small 4‑wire Ansatz."""
    def __init__(self) -> None:
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

# --------------------------------------------------------------------------- #
# 2. Default quantum model (Quantum‑NAT style)
# --------------------------------------------------------------------------- #

class QFCModel(tq.QuantumModule):
    """Quantum fully‑connected network inspired by the Quantum‑NAT paper."""
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
        pooled = torch.nn.functional.avg_pool2d(x, 6).view(bsz, 16)
        self.encoder(qdev, pooled)
        self.q_layer(qdev)
        out = self.measure(qdev)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 3. Hybrid estimator
# --------------------------------------------------------------------------- #

class HybridBaseEstimator:
    """Evaluate a TorchQuantum model and optionally a kernel.

    Parameters
    ----------
    model : tq.QuantumModule
        The circuit to evaluate.  If ``None`` a lightweight QFCModel is used.
    kernel : Callable | None
        Callable accepting two tensors and returning a scalar.  By default
        the quantum kernel defined above is used, with a classical RBF fallback.
    """
    def __init__(self,
                 model: tq.QuantumModule | None = None,
                 kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None) -> None:
        self.model = model or QFCModel()
        self.kernel = kernel or (lambda x, y: kernel_matrix([x], [y])[0, 0])

        # Keep a list of parameter names for binding
        self._parameters = [p.name for p in self.model.parameters()]

    # --------------------------------------------------------------------- #
    # Evaluate expectations
    # --------------------------------------------------------------------- #
    def _bind(self, param_values: Sequence[float]) -> tq.QuantumDevice:
        """Bind parameters to the underlying circuit."""
        if len(param_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, param_values))
        return self.model.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[tq.QuantumOperator],
                 parameter_sets: Sequence[Sequence[float]],
                 *,
                 shots: int | None = None,
                 seed: int | None = None) -> List[List[complex]]:
        """Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : iterable of quantum operators
        parameter_sets : sequence of parameter vectors
        shots : optional number of shots to sample from the quantum state
        """
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)

        if shots is None:
            return results

        rng = np.random.default_rng(seed)
        noisy: List[List[complex]] = []
        for row in results:
            noisy_row = [complex(rng.normal(val.real, 1 / shots) + 1j * rng.normal(val.imag, 1 / shots)) for val in row]
            noisy.append(noisy_row)
        return noisy

    # --------------------------------------------------------------------- #
    # Kernel matrix
    # --------------------------------------------------------------------- #
    def evaluate_kernels(self,
                         x: Sequence[torch.Tensor],
                         y: Sequence[torch.Tensor]) -> np.ndarray:
        """Return Gram matrix using the configured kernel."""
        return np.array([[self.kernel(a, b).item() for b in y] for a in x])

__all__ = ["HybridBaseEstimator", "Kernel", "kernel_matrix", "QFCModel"]
