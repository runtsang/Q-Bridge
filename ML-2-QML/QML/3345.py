import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

class PhotonicEncoder(tq.QuantumModule):
    """Encode classical data into photonic modes via a fixed circuit."""
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for i in range(self.n_wires):
            func_name_dict["dgate"](q_device, wires=i, params=x[:, i])

class KernalAnsatz(tq.QuantumModule):
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

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel using a fixed ansatz."""
    def __init__(self, support_vectors: torch.Tensor):
        super().__init__()
        self.support_vectors = support_vectors
        self.n_wires = support_vectors.shape[1]
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [{"input_idx": [0], "func": "ry", "wires": [0]},
             {"input_idx": [1], "func": "ry", "wires": [1]},
             {"input_idx": [2], "func": "ry", "wires": [2]},
             {"input_idx": [3], "func": "ry", "wires": [3]}]
        )

    def kernel_value(self, x: torch.Tensor) -> torch.Tensor:
        # compute kernel between x and each support vector
        values = []
        for sv in self.support_vectors:
            self.ansatz(self.q_device, x, sv.unsqueeze(0))
            state = self.q_device.states.view(-1)
            values.append(torch.abs(state[0]))
        return torch.stack(values, dim=1)

class FraudDetectionHybrid(tq.QuantumModule):
    """Hybrid quantum model combining photonic encoding and quantum kernel."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 support_vectors: torch.Tensor):
        super().__init__()
        # Build photonic program
        self.program = sf.Program(2)
        with self.program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        self.simulator = sf.Simulator()
        # Quantum kernel
        self.kernel = QuantumKernel(support_vectors)
        # Classical classifier weights
        self.classifier_weights = torch.nn.Parameter(torch.randn(support_vectors.shape[0], 1))
        self.classifier_bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Run photonic circuit
        self.program.input = x.numpy()
        result = self.simulator.run(self.program)
        # Use the statevector as feature vector
        features = torch.tensor(result.statevector.real, dtype=torch.float32).unsqueeze(0)
        # Compute quantum kernel against support vectors
        kernel_vals = self.kernel.kernel_value(features)
        logits = kernel_vals @ self.classifier_weights + self.classifier_bias
        return logits

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
