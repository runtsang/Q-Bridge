"""
Hybrid binary classifier – quantum implementation.

Key components:
  * `QuantumAutoencoder` – SamplerQNN that implements the auto‑encoder
    circuit from Reference 2.
  * `QuantumClassifier` – EstimatorQNN that evaluates a layered ansatz
    from Reference 3 and returns a 2‑class probability vector.
  * `HybridFunctionQML` – torch.autograd.Function that bridges the
    quantum output to the PyTorch graph.
  * `HybridBinaryClassifierQML` – the main model that mirrors the
    classical counterpart but swaps the dense head for a quantum
    expectation head.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import ParameterVector, Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN

# --------------------------------------------------------------------------- #
# 1. Quantum Auto‑Encoder (from Reference 2)
# --------------------------------------------------------------------------- #
def _build_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Construct the auto‑encoder circuit used in the reference seed.
    The circuit performs a RealAmplitudes ansatz, a swap‑test, and a measurement.
    """
    qr = QuantumCircuit(num_latent + 2 * num_trash + 1)
    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qr.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    qr.barrier()
    # Swap‑test
    aux = num_latent + 2 * num_trash
    qr.h(aux)
    for i in range(num_trash):
        qr.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qr.h(aux)
    qr.measure(aux, 0)
    return qr


def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Return a SamplerQNN that implements the quantum auto‑encoder."""
    qc = _build_autoencoder_circuit(num_latent, num_trash)
    sampler = StatevectorSampler()
    # No input parameters, weight params are the ansatz angles
    weight_params = qc.parameters
    return SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=weight_params,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )


# --------------------------------------------------------------------------- #
# 2. Quantum Classifier (from Reference 3)
# --------------------------------------------------------------------------- #
def _build_classifier_circuit(num_qubits: int, depth: int) -> QuantumCircuit:
    """
    Construct a layered ansatz with explicit encoding and variational
    parameters.  This is the quantum analogue of the feed‑forward
    classifier in the classical seed.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)

    return qc


def QuantumClassifier(num_qubits: int = 1, depth: int = 3) -> EstimatorQNN:
    """
    Return an EstimatorQNN that evaluates the layered ansatz and
    returns a 2‑class probability vector.
    """
    qc = _build_classifier_circuit(num_qubits, depth)
    # Observables: Z on each qubit (here only one qubit)
    observables = [SparsePauliOp.from_list([("Z", 1)])]
    estimator = StatevectorEstimator()
    return EstimatorQNN(
        circuit=qc,
        observables=observables,
        input_params=[],
        weight_params=qc.parameters,
        estimator=estimator,
    )


# --------------------------------------------------------------------------- #
# 3. Hybrid Function – bridges quantum output to PyTorch
# --------------------------------------------------------------------------- #
class HybridFunctionQML(torch.autograd.Function):
    """
    Differentiable wrapper that runs a quantum circuit via EstimatorQNN
    or SamplerQNN and returns a torch tensor.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, qnn: nn.Module, shift: float = 0.0) -> torch.Tensor:
        # Store for backward
        ctx.qnn = qnn
        ctx.shift = shift

        # Convert to numpy
        np_inputs = inputs.detach().cpu().numpy().astype(np.float32)
        # Quantum evaluation
        if isinstance(qnn, EstimatorQNN):
            # EstimatorQNN expects a 1‑D array of parameters
            outputs = qnn(np_inputs)
        else:
            outputs = qnn(np_inputs)
        # Convert back to torch
        torch_outputs = torch.tensor(outputs, dtype=torch.float32, device=inputs.device)
        ctx.save_for_backward(torch_outputs)
        return torch_outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # For simplicity, use a finite‑difference estimate
        # (no analytical gradient from the quantum backend).
        inputs, = ctx.saved_tensors
        shift = ctx.shift
        grad = torch.zeros_like(inputs)
        eps = 1e-3
        for i in range(inputs.numel()):
            idx = np.unravel_index(i, inputs.shape)
            perturbed = inputs.detach().cpu().numpy().astype(np.float32)
            perturbed[idx] += eps
            out_plus = ctx.qnn(perturbed)
            perturbed[idx] -= 2 * eps
            out_minus = ctx.qnn(perturbed)
            grad[idx] = (out_plus - out_minus) / (2 * eps)
        grad = grad.to(inputs.device)
        return grad * grad_output, None, None


# --------------------------------------------------------------------------- #
# 4. Convolutional Backbone (shared)
# --------------------------------------------------------------------------- #
class ConvBackbone(nn.Module):
    """Standard 2‑D CNN backbone (identical to the ML version)."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return self.drop2(x)


# --------------------------------------------------------------------------- #
# 5. Main Hybrid Binary Classifier – Quantum Path
# --------------------------------------------------------------------------- #
class HybridBinaryClassifierQML(nn.Module):
    """
    A hybrid quantum‑classical binary classifier that mirrors the
    classical implementation but substitutes the dense head with a
    variational quantum circuit.

    Parameters
    ----------
    use_quantum_autoencoder : bool
        If True, prepend a quantum auto‑encoder (SamplerQNN) before the head.
    use_quantum_head : bool
        If True, the final head uses a quantum EstimatorQNN.
    autoencoder_params : dict | None
        Parameters for the quantum auto‑encoder (num_latent, num_trash).
    classifier_params : dict | None
        Parameters for the quantum classifier (num_qubits, depth).
    """
    def __init__(
        self,
        *,
        use_quantum_autoencoder: bool = False,
        use_quantum_head: bool = True,
        autoencoder_params: Optional[dict] = None,
        classifier_params: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        feature_dim = 55815  # flattened output size
        self.fc1 = nn.Linear(feature_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.use_quantum_autoencoder = use_quantum_autoencoder
        if use_quantum_autoencoder:
            params = autoencoder_params or {"num_latent": 3, "num_trash": 2}
            self.autoencoder = QuantumAutoencoder(**params)
        else:
            self.autoencoder = None

        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            params = classifier_params or {"num_qubits": 1, "depth": 3}
            self.head_qnn = QuantumClassifier(**params)
            # Wrap in hybrid function for autograd
            self.head = HybridFunctionQML.apply
        else:
            # Fallback to a simple classical sigmoid head
            self.head = lambda x: torch.sigmoid(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.backbone(inputs)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.use_quantum_autoencoder:
            # Quantum auto‑encoder expects a 1‑D array; flatten and run
            flat = x.view(-1).cpu().numpy()
            ae_output = self.autoencoder(flat)
            # Convert back to torch
            x = torch.tensor(ae_output, dtype=torch.float32, device=x.device)

        if self.use_quantum_head:
            logits = self.head(x, self.head_qnn, shift=0.0)
        else:
            logits = self.head(x)

        probs = torch.cat((logits, 1 - logits), dim=-1)
        return probs


# --------------------------------------------------------------------------- #
# 6. Public API
# --------------------------------------------------------------------------- #
__all__ = [
    "QuantumAutoencoder",
    "QuantumClassifier",
    "HybridBinaryClassifierQML",
]
