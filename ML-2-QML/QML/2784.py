import torch
import torchquantum as tq
import numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from torchquantum.functional import func_name_dict
from typing import Iterable, Sequence
from dataclasses import dataclass

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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

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

class KernalAnsatz(tq.QuantumModule):
    """Quantum kernel ansatz using a list of parameterised gates."""
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
    """Variational quantum kernel evaluated on a fixed ansatz."""
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

class HybridQuantumKernelFraudDetector(tq.QuantumModule):
    """Quantum‑centric hybrid model combining a variational kernel and a photonic fraud circuit."""
    def __init__(self):
        super().__init__()
        self.kernel = Kernel()
        # Photonic fraud detection program
        self.fraud_prog = build_fraud_detection_program(
            FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            [],
        )
        self.fraud_sim = sf.Simulator("gaussian")

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        Returns a tuple (quantum_kernel, photonic_output).
        """
        q_kernel = self.kernel(x, y)
        photonic_out = self._run_fraud_circuit(x)
        return q_kernel, photonic_out

    def _run_fraud_circuit(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Simulate the photonic fraud detection circuit for each sample.
        For simplicity, encode the first two features as displacements on the two modes.
        """
        results = []
        for sample in inputs:
            self.fraud_sim.reset()
            # Apply the fixed program
            self.fraud_sim.apply(self.fraud_prog)
            # Extract photon‑number expectation of mode 0
            state = self.fraud_sim.state
            exp = state.expectation_value(sf.ops.Fock(0))
            results.append(exp)
        return torch.tensor(results)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "KernalAnsatz",
    "Kernel",
    "HybridQuantumKernelFraudDetector",
]
