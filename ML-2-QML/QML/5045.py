"""UnifiedClassifier – quantum‑centric implementation.

The module mirrors the classical interface but replaces the head with a
parameterised quantum circuit that can be run on Qiskit or Strawberry Fields.
The same head types are supported; the quantum module internally builds the
circuit and exposes a PyTorch autograd wrapper so that the whole network can
be trained end‑to‑end with gradient‑based optimisers.

Typical usage:
    clf = UnifiedClassifier(head_type="sampler")
    preds = clf(torch.randn(8, 3, 32, 32))
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List, Dict, Any

# --------------------------------------------------------------------------- #
# 1.  Quantum helpers – wrappers around Qiskit/Strawberry Fields
# --------------------------------------------------------------------------- #

import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Sampler as QiskitSampler

import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# 2.  Generic quantum circuit wrapper
# --------------------------------------------------------------------------- #

class QuantumCircuitWrapper:
    """Executes a parameterised Qiskit circuit and returns expectation values."""
    def __init__(self, circuit: qiskit.QuantumCircuit, backend, shots: int = 1024):
        self.circuit = circuit
        self.backend = backend
        self.shots = shots

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.circuit.parameters[i]: th} for i, th in enumerate(thetas)],
        )
        job = self.backend.run(qobj)
        result = job.result()
        # Expectation of Z for each qubit
        expectations = []
        for idx in range(self.circuit.num_qubits):
            counts = result.get_counts(idx)
            exp = 0.0
            for bitstring, cnt in counts.items():
                state = int(bitstring[::-1], 2)  # little‑endian
                prob = cnt / self.shots
                exp += (1 - 2 * state) * prob
            expectations.append(exp)
        return np.array(expectations)

# --------------------------------------------------------------------------- #
# 3.  Hybrid autograd function
# --------------------------------------------------------------------------- #

class HybridFunctionQuantum(torch.autograd.Function):
    """Differentiable wrapper that executes a quantum circuit."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, wrapper: QuantumCircuitWrapper, shift: float):
        ctx.wrapper = wrapper
        ctx.shift = shift
        inputs_np = inputs.detach().cpu().numpy()
        expectation_z = ctx.wrapper.run(inputs_np)
        result = torch.tensor(expectation_z, device=inputs.device, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.ones_like(inputs.numpy()) * ctx.shift
        gradients = []
        for idx, val in enumerate(inputs.numpy()):
            right = ctx.wrapper.run([val + shift[idx]])
            left = ctx.wrapper.run([val - shift[idx]])
            gradients.append(right - left)
        gradients = torch.tensor(gradients, dtype=torch.float32, device=inputs.device)
        return gradients * grad_output, None, None

# --------------------------------------------------------------------------- #
# 4.  Head implementations
# --------------------------------------------------------------------------- #

class QuantumHead(nn.Module):
    """Hybrid head that forwards activations through a Qiskit circuit."""
    def __init__(self, circuit_builder, num_qubits: int, depth: int = 1,
                 shots: int = 1024, shift: float = np.pi / 2) -> None:
        super().__init__()
        circuit, _, _, _ = circuit_builder(num_qubits, depth)
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.wrapper = QuantumCircuitWrapper(circuit, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expectation values are returned as a column vector
        return HybridFunctionQuantum.apply(x, self.wrapper, self.shift).unsqueeze(-1)


# -------------------------------- Sampler head --------------------------------

def sampler_circuit_builder(num_qubits: int, depth: int = 1) -> Tuple[qiskit.QuantumCircuit, List[Any], List[Any], List[Any]]:
    """Build a simple two‑qubit sampler circuit as in Pair 2."""
    qc = qiskit.QuantumCircuit(num_qubits)
    inputs = ParameterVector("input", num_qubits)
    weights = ParameterVector("weight", num_qubits * depth)

    for i in range(num_qubits):
        qc.ry(inputs[i], i)
    for i in range(num_qubits):
        qc.rz(weights[i], i)
    return qc, list(inputs), list(weights), []

class SamplerHead(QuantumHead):
    def __init__(self, num_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__(sampler_circuit_builder, num_qubits, depth=1,
                         shots=shots, shift=shift)


# -------------------------------- Fraud‑detection head --------------------------------

def fraud_circuit_builder(num_qubits: int, depth: int = 1) -> Tuple[Program, List[Any], List[Any], List[Any]]:
    """Create a Strawberry Fields program mirroring the photonic fraud detection circuit."""
    prog = Program(num_qubits)
    q = prog.qubits
    # The same parameters as in the classical counterpart
    params = FraudLayerParameters(
        bs_theta=0.0, bs_phi=0.0,
        phases=(0.0, 0.0),
        squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
        displacement_r=(0.0, 0.0), displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0)
    )

    _apply_layer(q, params, clip=False)
    return prog, [], [], []

def _apply_layer(modes: Iterable[Program], params: FraudLayerParameters, *, clip: bool) -> None:
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

class FraudHead(QuantumHead):
    def __init__(self, num_qubits: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__(fraud_circuit_builder, num_qubits, depth=1,
                         shots=shots, shift=shift)


# -------------------------------- Classifier head --------------------------------

def quantum_classifier_circuit_builder(num_qubits: int, depth: int = 2) -> Tuple[qiskit.QuantumCircuit, List[Any], List[Any], List[SparsePauliOp]]:
    """Return a layered ansatz with encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = qiskit.QuantumCircuit(num_qubits)
    for idx, qubit in enumerate(range(num_qubits)):
        qc.rx(encoding[idx], qubit)

    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[qubit], qubit)
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
                   for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class ClassifierHead(QuantumHead):
    def __init__(self, num_qubits: int = 10, depth: int = 2, shots: int = 1024, shift: float = np.pi / 2):
        super().__init__(quantum_classifier_circuit_builder, num_qubits,
                         depth=depth, shots=shots, shift=shift)


# --------------------------------------------------------------------------- #
# 5.  UnifiedClassifier – quantum version
# --------------------------------------------------------------------------- #

class UnifiedClassifier(nn.Module):
    """CNN‑based classifier with a quantum head.

    Parameters
    ----------
    head_type : str
        One of ``'quantum'``, ``'sampler'``, ``'fraud'`` or ``'classifier'``.
    """
    def __init__(self, head_type: str = "quantum", **head_kwargs) -> None:
        super().__init__()
        self.head_type = head_type
        self.base = self._build_base()
        self.head = self._build_head(head_type, **head_kwargs)

    # --------------------- base CNN ---------------------------------------

    def _build_base(self) -> nn.Module:
        conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        pool = nn.MaxPool2d(kernel_size=2, stride=1)
        drop1 = nn.Dropout2d(p=0.2)
        drop2 = nn.Dropout2d(p=0.5)
        fc1 = nn.Linear(55815, 120)
        fc2 = nn.Linear(120, 84)
        fc3 = nn.Linear(84, 1)

        return nn.Sequential(
            conv1, nn.ReLU(), pool, drop1,
            conv2, nn.ReLU(), pool, drop1,
            nn.Flatten(), fc1, nn.ReLU(), drop2,
            fc2, nn.ReLU(), fc3
        )

    # --------------------- head construction --------------------------------

    def _build_head(self, head_type: str, **kwargs) -> nn.Module:
        if head_type == "quantum":
            return QuantumHead(**kwargs)
        if head_type == "sampler":
            return SamplerHead(**kwargs)
        if head_type == "fraud":
            return FraudHead(**kwargs)
        if head_type == "classifier":
            return ClassifierHead(**kwargs)
        raise ValueError(f"Unsupported head_type: {head_type}")

    # --------------------- forward -----------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        logits = self.head(features)
        probs = F.softmax(logits, dim=-1)
        return probs


__all__ = ["UnifiedClassifier", "QuantumHead", "HybridFunctionQuantum"]
