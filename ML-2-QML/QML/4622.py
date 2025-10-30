"""Hybrid kernel estimator with quantum kernels and a variational quantum neural network."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from torchquantum.functional import func_name_dict


class QuantumKernelAnsatz(tq.QuantumModule):
    """Parametrised encoding of a pair of classical feature vectors."""

    def __init__(self, func_list: list[dict]) -> None:
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


class QuantumKernel(tq.QuantumModule):
    """Quantum kernel evaluated via a fixed TorchQuantum ansatz."""

    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = QuantumKernelAnsatz(
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


class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Random two‑qubit quantum kernel applied to 2×2 image patches."""

    def __init__(self) -> None:
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
        return torch.cat(patches, dim=1)  # (bsz, 4*14*14)


class QuantumEstimatorQNN(tq.QuantumModule):
    """Small variational quantum neural network."""

    def __init__(self, n_qubits: int) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.q_device = tq.QuantumDevice(n_wires=n_qubits)
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Parameterised read‑out weights
        self.params = nn.Parameter(torch.randn(n_qubits))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        self.q_device.reset_states(bsz)
        # Encode each feature with an Ry rotation
        for i in range(self.n_qubits):
            tq.operators.RY(self.q_device, wires=[i], params=x[:, i])
        # Entangle
        for i in range(self.n_qubits - 1):
            tq.operators.CNOT(self.q_device, wires=[i, i + 1])
        # Readout
        meas = self.measure(self.q_device)  # (bsz, n_qubits)
        return torch.matmul(meas, self.params)  # (bsz,)


class HybridKernelEstimator(tq.QuantumModule):
    """
    Quantum kernel ridge regression with an optional quanvolution front‑end and a
    variational quantum neural network that maps kernel rows to predictions.
    """

    def __init__(
        self,
        use_quanvolution: bool = False,
        gamma: float = 1.0,
        hidden_sizes: list[int] | tuple[int,...] = (8, 4),
        regularization: float = 1e-5,
    ) -> None:
        super().__init__()
        self.use_quanvolution = use_quanvolution
        self.kernel = QuantumKernel(gamma)
        self.hidden_sizes = list(hidden_sizes)
        self.regularization = regularization
        self.alpha: torch.Tensor | None = None
        self.n_train: int | None = None
        self.X_train: torch.Tensor | None = None
        self.estimator_qnn: QuantumEstimatorQNN | None = None
        if self.use_quanvolution:
            self.qfilter = QuantumQuanvolutionFilter()

    def kernel_matrix(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Construct pairwise kernel matrix by explicit evaluation."""
        N_a, N_b = a.shape[0], b.shape[0]
        K = torch.empty((N_a, N_b), device=a.device)
        for i in range(N_a):
            for j in range(N_b):
                K[i, j] = self.kernel(a[i], b[j])
        return K

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        if self.use_quanvolution:
            X = self.qfilter(X)
        K_train = self.kernel_matrix(X, X)
        K_reg = K_train + self.regularization * torch.eye(K_train.size(0), device=K_train.device)
        self.alpha = torch.linalg.solve(K_reg, y.squeeze())
        self.n_train = X.size(0)
        self.X_train = X
        # Instantiate the variational QNN that maps a full kernel row to a scalar
        self.estimator_qnn = QuantumEstimatorQNN(self.n_train)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_quanvolution:
            X = self.qfilter(X)
        K_test = self.kernel_matrix(X, self.X_train)
        return self.estimator_qnn(K_test)


__all__ = ["HybridKernelEstimator"]
