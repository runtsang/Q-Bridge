"""Hybrid fraud‑detection model that combines classical and quantum building blocks.

The model is a drop‑in replacement for the original :class:`FraudDetection.py` seed while
incorporating ideas from the quantum kernel, Quantum‑NAT, and SamplerQNN seeds.
It is fully importable as ``FraudDetectionHybrid`` and exposes a single
``forward`` method that accepts a batch of 2‑D tensors (e.g. flattened images)
and returns a probability distribution over two classes.

The architecture is:

1. **CNN feature extractor** – a lightweight 2‑D network inspired by
   :class:`QFCModel` from the Quantum‑NAT seed.
2. **Quantum kernel** – a TorchQuantum module that evaluates an RBF‑style
   similarity between the CNN features and a learnable reference vector.
3. **Photonic circuit** – a Strawberry‑Fields program that takes the first
   two CNN features as displacement parameters and returns the amplitude
   of the first mode of the output state.  The amplitude is modulated by the
   kernel score.
4. **Quantum sampler** – a simple Qiskit sampler that maps the modulated
   amplitude to a 2‑class probability distribution.

The module is intentionally lightweight so that it can be trained on
CPU‑only machines, yet it demonstrates how classical and quantum
components can be orchestrated in a single forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields import Engine
from torchquantum.functional import func_name_dict
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 1. Classical CNN feature extractor (Quantum‑NAT inspired)
# --------------------------------------------------------------------------- #
class _CNNFeatureExtractor(nn.Module):
    """Extracts 4‑dimensional features from a 1‑channel image."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feats = self.features(x)
        flattened = feats.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# --------------------------------------------------------------------------- #
# 2. Quantum kernel module (TorchQuantum)
# --------------------------------------------------------------------------- #
class _KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors using a fixed list of single‑qubit rotations."""
    def __init__(self, func_list: List[dict]) -> None:
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

class _Kernel(tq.QuantumModule):
    """Quantum kernel that returns the absolute value of the first amplitude."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = _KernalAnsatz(
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

# --------------------------------------------------------------------------- #
# 3. Photonic circuit (Strawberry‑Fields)
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic layer (mirrors the seed)."""
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

def _apply_layer(modes: Iterable, params: FraudLayerParameters, *, clip: bool) -> None:
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

# --------------------------------------------------------------------------- #
# 4. Quantum sampler (Qiskit)
# --------------------------------------------------------------------------- #
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

def _create_sampler_qnn() -> SamplerQNN:
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler = Sampler()
    return SamplerQNN(circuit=qc, input_params=inputs,
                      weight_params=weights, sampler=sampler)

# --------------------------------------------------------------------------- #
# 5. Main hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that integrates classical CNN, quantum kernel,
    photonic circuit, and quantum sampler.

    Parameters
    ----------
    reference_vector : torch.Tensor
        Learnable 4‑dimensional vector used by the quantum kernel.
    """
    def __init__(self, reference_vector: torch.Tensor | None = None) -> None:
        super().__init__()
        self.cnn = _CNNFeatureExtractor()
        self.kernel = _Kernel()
        # Initialize a random reference vector if not provided
        self.register_parameter(
            "reference_vector",
            nn.Parameter(reference_vector if reference_vector is not None
                         else torch.randn(4))
        )
        # Photonic circuit parameters: we use a fixed trivial layer
        self.photonic_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.1,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        self.photonic_program = build_fraud_detection_program(
            self.photonic_params, layers=[]
        )
        self.sampler = _create_sampler_qnn()
        self.backend = Engine('fock', backend_options={"cutoff_dim": 10})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Probabilities of shape (N, 2).
        """
        # 1. Classical CNN features
        feats = self.cnn(x)          # (N, 4)

        # 2. Quantum kernel similarity to reference vector
        kernel_score = self.kernel(feats, self.reference_vector)  # scalar

        # 3. Photonic circuit: use first two features as displacement amplitudes
        #    (we overwrite displacement_r in the program)
        self.photonic_params.displacement_r = (float(feats[:, 0].mean()),
                                               float(feats[:, 1].mean()))
        self.photonic_program = build_fraud_detection_program(
            self.photonic_params, layers=[]
        )
        prog = self.photonic_program
        with self.backend as eng:
            eng.run(prog)
            state = eng.backend.state
        # amplitude of first mode
        amp = torch.tensor(abs(state[0]), dtype=torch.float32)

        # 4. Modulate amplitude with kernel score
        modulated = amp * kernel_score

        # 5. Quantum sampler to produce two‑class probabilities
        #    We feed the modulated amplitude as a single input parameter.
        #    The sampler expects a 2‑dimensional input vector, so we broadcast.
        sampler_input = torch.stack([modulated, modulated], dim=1)
        probs = self.sampler(sampler_input)
        return probs

__all__ = ["FraudDetectionHybrid"]
