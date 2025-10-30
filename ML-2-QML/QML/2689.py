"""
FraudDetectionHybrid: Quantum‑backed photonic circuit with hybrid expectation layer.
"""

import numpy as np
import torch
from torch import nn

import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate

import qiskit
from qiskit import assemble, transpile
from qiskit.providers import Backend

# --------------------------------------------------------------------------- #
# 1. Photonic backbone
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))

class FraudLayerParameters:
    """Parameters for a single photonic layer."""
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

class QuantumPhotonicCircuit:
    """
    Photonic circuit that mirrors the classical photonic‑style layers.
    It encodes a 2‑dimensional input via displacement gates and then
    applies a sequence of beam‑splitter, squeezing, phase, displacement
    and Kerr gates.
    """

    def __init__(self, input_params, layers):
        self.input_params = input_params
        self.layers = layers
        # Fock backend with modest cutoff for simulation
        self.backend = sf.backends.Fock(l=10)

    def _apply_layer(self, q, params, clip):
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else _clip(r, 5), phi) | q[i]
        BSgate(params.bs_theta, params.bs_phi) | (q[0], q[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | q[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else _clip(r, 5), phi) | q[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else _clip(k, 1)) | q[i]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode the 2‑dimensional input as displacements and run the
        photonic circuit.  Returns the expectation value of the photon
        number operator for each mode as a 2‑element tensor.
        """
        program = sf.Program(2)
        with program.context as q:
            # Encode data
            Dgate(inputs[0], 0) | q[0]
            Dgate(inputs[1], 0) | q[1]
            # Apply layers
            self._apply_layer(q, self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)

        result = self.backend.run(program).state
        n0 = result.expectation_value(sf.ops.Number(0))
        n1 = result.expectation_value(sf.ops.Number(1))
        return torch.tensor([n0, n1], dtype=torch.float32)

# --------------------------------------------------------------------------- #
# 2. Quantum expectation head
# --------------------------------------------------------------------------- #
class QuantumCircuit:
    """
    Parametrised one‑qubit circuit used as the expectation head.
    The circuit consists of a single Ry rotation followed by a
    measurement in the Z basis.  The expectation value of Z is
    cos(theta).
    """

    def __init__(self, backend, shots=100):
        self.backend = backend
        self.shots = shots
        self.circuit = qiskit.QuantumCircuit(1)
        self.theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(0)
        self.circuit.ry(self.theta, 0)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: theta} for theta in thetas],
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(count_dict):
            counts = np.array(list(count_dict.values()))
            states = np.array(list(count_dict.keys())).astype(float)
            probabilities = counts / self.shots
            return np.sum(states * probabilities)

        if isinstance(result, list):
            return np.array([expectation(item) for item in result])
        return np.array([expectation(result)])

class HybridFunction(torch.autograd.Function):
    """
    Differentiable bridge between PyTorch and the quantum expectation head.
    """

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, circuit: QuantumCircuit, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.quantum_circuit = circuit
        expectation_z = ctx.quantum_circuit.run(inputs.tolist())
        result = torch.tensor(expectation_z, dtype=torch.float32)
        ctx.save_for_backward(inputs, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, _ = ctx.saved_tensors
        shift = np.full_like(inputs.numpy(), ctx.shift)
        gradients = []
        for idx, value in enumerate(inputs.numpy()):
            expectation_right = ctx.quantum_circuit.run([value + shift[idx]])[0]
            expectation_left = ctx.quantum_circuit.run([value - shift[idx]])[0]
            gradients.append(expectation_right - expectation_left)
        gradients = torch.tensor(gradients, dtype=torch.float32)
        return gradients * grad_output, None, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a quantum circuit.
    """

    def __init__(self, backend, shots=100, shift=0.0):
        super().__init__()
        self.quantum_circuit = QuantumCircuit(backend, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
        return HybridFunction.apply(squeezed, self.quantum_circuit, self.shift)

# --------------------------------------------------------------------------- #
# 3. Full hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    Full hybrid fraud‑detection model that combines a photonic backbone
    with a quantum expectation head.  The backbone processes the raw
    2‑dimensional input, while the head refines the decision using a
    parameterised quantum circuit.
    """

    def __init__(self, backbone_params, head_shift=0.0):
        super().__init__()
        # backbone_params is an iterable of FraudLayerParameters
        self.backbone = QuantumPhotonicCircuit(backbone_params[0], backbone_params[1:])
        backend = qiskit.Aer.get_backend("aer_simulator")
        self.hybrid = Hybrid(backend, shots=100, shift=head_shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: inputs is a 2‑dimensional tensor.
        """
        features = self.backbone.forward(inputs)
        # The hybrid expects a 1‑dimensional tensor of shape (batch, in_features)
        # We treat the 2‑element feature vector as a single batch sample.
        features = features.unsqueeze(0)
        output = self.hybrid.forward(features)
        # Convert to probability vector [p, 1-p]
        p = output.squeeze(-1)
        return torch.cat((p, 1 - p), dim=-1)

__all__ = ["FraudDetectionHybrid", "QuantumPhotonicCircuit", "Hybrid", "QuantumCircuit", "HybridFunction"]
