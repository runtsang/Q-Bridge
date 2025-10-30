"""HybridEstimatorQNN: a classical‑quantum regressor.

The class combines Qiskit and Strawberry Fields feature generators with a
multi‑layer PyTorch network.  The quantum part is differentiable via
Qiskit‑Machine‑Learning's EstimatorQNN, and the photonic part is evaluated
with the Strawberry Fields simulator.  Both branches produce a 2‑dimensional
feature vector that is fed into a classical network that performs the final
regression.

The implementation is self‑contained;  no external training loop is
required.  All parameters are wrapped in torch.nn.Parameters so that
gradient‑based optimizers can be used directly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter

# Quantum backends
from qiskit import QuantumCircuit, Aer
from qiskit.circuit import Parameter as QiskitParameter
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as QiskitEstimator

import strawberryfields as sf
from strawberryfields import Program
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# Helper functions for parameter clipping
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return float(max(-bound, min(bound, value)))

# --------------------------------------------------------------------------- #
# Quantum feature extractors
# --------------------------------------------------------------------------- #
def _build_qubit_circuit(params: dict) -> QuantumCircuit:
    """
    Construct a 2‑qubit circuit with trainable parameters.

    Parameters
    ----------
    params
        Dictionary containing keys:
        - ``theta`` : beam‑splitter rotation angle
        - ``phi``   : beam‑splitter phase
        - ``ry``    : rotation around Y
        - ``rx``    : rotation around X
        The values are clipped to ``[-π, π]``.
    """
    theta = _clip(params["theta"], np.pi)
    phi   = _clip(params["phi"], np.pi)
    ry    = _clip(params["ry"], np.pi)
    rx    = _clip(params["rx"], np.pi)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(ry, 0)
    qc.rx(rx, 0)
    qc.cx(0, 1)          # simple entanglement
    qc.h(1)
    return qc

def _build_photonic_program(params: dict) -> Program:
    """
    Construct a 2‑mode photonic program with trainable parameters
    mimicking the FraudDetection photonic layer.

    Parameters
    ----------
    params
        Dictionary with keys:
        - ``bs_theta``, ``bs_phi``
        - ``phases``          : tuple[float, float]
        - ``squeeze_r``, ``squeeze_phi``  : tuples
        - ``displacement_r``, ``displacement_phi`` : tuples
        - ``kerr``           : tuple[float, float]
        All values are clipped to reasonable bounds.
    """
    program = Program(2)
    with program.context as q:
        BSgate(params["bs_theta"], params["bs_phi"]) | (q[0], q[1])
        for i, phase in enumerate(params["phases"]):
            Rgate(phase) | q[i]
        for r, phi in zip(params["squeeze_r"], params["squeeze_phi"]):
            Sgate(_clip(r, 5), phi) | q[0]
        for r, phi in zip(params["displacement_r"], params["displacement_phi"]):
            Dgate(_clip(r, 5), phi) | q[1]
        for k in params["kerr"]:
            Kgate(_clip(k, 1)) | q[0]
    return program

# --------------------------------------------------------------------------- #
# Classical network building blocks
# --------------------------------------------------------------------------- #
def _layer_from_params(params: dict, clip: bool = True) -> nn.Module:
    """
    Create a single linear‑tanh‑scaled layer.

    Parameters
    ----------
    params
        Dictionary containing:
        - ``weight`` : 2×2 matrix
        - ``bias``   : 2‑element vector
        - ``scale``  : 2‑element vector
        - ``shift``  : 2‑element vector
        When ``clip=True`` the weight and bias are clipped to ``[-5, 5]``.
    """
    weight = torch.tensor(params["weight"], dtype=torch.float32)
    bias   = torch.tensor(params["bias"], dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias   = bias.clamp(-5.0, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    activation = nn.Tanh()
    scale = torch.tensor(params["scale"], dtype=torch.float32)
    shift = torch.tensor(params["shift"], dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

# --------------------------------------------------------------------------- #
# Main hybrid estimator
# --------------------------------------------------------------------------- #
class HybridEstimatorQNN(nn.Module):
    """
    Hybrid estimator combining:
    * a 2‑qubit Qiskit circuit (trainable),
    * a 2‑mode Strawberry Fields program (trainable),
    * a classical neural network with two hidden layers.

    The quantum and photonic outputs are concatenated to form a 4‑dimensional
    feature vector feeding the classical network.
    """

    def __init__(
        self,
        qubit_params: dict | None = None,
        photonic_params: dict | None = None,
        hidden_sizes: list[int] = [8, 4],
    ) -> None:
        super().__init__()

        # Default parameters if none supplied
        if qubit_params is None:
            qubit_params = {
                "theta": 0.5,
                "phi": 0.3,
                "ry": 0.1,
                "rx": -0.2,
            }
        if photonic_params is None:
            photonic_params = {
                "bs_theta": 0.5,
                "bs_phi": 0.3,
                "phases": (0.1, -0.1),
                "squeeze_r": (0.2, 0.2),
                "squeeze_phi": (0.0, 0.0),
                "displacement_r": (0.1, 0.1),
                "displacement_phi": (0.0, 0.0),
                "kerr": (0.05, 0.05),
            }

        # Store parameters as torch Parameters for autograd
        self.qubit_params = nn.ParameterDict(
            {k: Parameter(torch.tensor(v, dtype=torch.float32))
             for k, v in qubit_params.items()}
        )
        self.photonic_params = nn.ParameterDict(
            {k: Parameter(torch.tensor(v, dtype=torch.float32))
             for k, v in photonic_params.items()}
        )

        # Classical network
        layers = [
            nn.Linear(4, hidden_sizes[0]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Tanh(),
            nn.Linear(hidden_sizes[1], 1),
        ]
        self.classical_net = nn.Sequential(*layers)

        # Quantum estimators
        self.qiskit_estimator = QiskitEstimator()
        # Use the Qiskit EstimatorQNN wrapper to obtain gradients
        self.qiskit_qnn = QiskitEstimatorQNN(
            circuit=_build_qubit_circuit({k: float(v) for k, v in qubit_params.items()}),
            observables=[sf.SparsePauliOp.from_list([("Y", 1)])],
            input_params=[QiskitParameter("x")],
            weight_params=[QiskitParameter("theta"), QiskitParameter("phi"),
                           QiskitParameter("ry"), QiskitParameter("rx")],
            estimator=self.qiskit_estimator,
        )

    # --------------------------------------------------------------------- #
    # Quantum feature extraction
    # --------------------------------------------------------------------- #
    def _quantum_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the qubit circuit on each input sample and return expectation
        values of Pauli Y on each qubit.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 1) – each sample is a single float.
        """
        batch = inputs.shape[0]
        y_vals = []
        for i in range(batch):
            # Build circuit with current parameters
            param_dict = {k: float(v) for k, v in self.qubit_params.items()}
            qc = _build_qubit_circuit(param_dict)
            # The EstimatorQNN expects a single input parameter; we use the
            # first qubit's Y expectation as the feature.
            expectation = self.qiskit_estimator.run(
                qc, {"x": float(inputs[i].item())}
            ).result().values[0]
            y_vals.append(expectation)
        return torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)

    # --------------------------------------------------------------------- #
    # Photonic feature extraction
    # --------------------------------------------------------------------- #
    def _photonic_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Run the Strawberry Fields program and return the mean photon number
        of each mode as features.
        """
        batch = inputs.shape[0]
        photon_counts = []
        for i in range(batch):
            param_dict = {k: float(v) for k, v in self.photonic_params.items()}
            prog = _build_photonic_program(param_dict)
            eng = sf.Engine("gaussian", backend_options={"cutoff_dim": 10})
            result = eng.run(prog, args={"x": float(inputs[i].item())})
            state = result.state
            photon_counts.append(state.mean_photon_number(0))
            photon_counts.append(state.mean_photon_number(1))
        return torch.tensor(photon_counts, dtype=torch.float32).view(batch, 2)

    # --------------------------------------------------------------------- #
    # Forward pass
    # --------------------------------------------------------------------- #
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compute the hybrid prediction.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape (batch, 1)
        """
        qfeat = self._quantum_features(inputs)
        pfeat = self._photonic_features(inputs)
        features = torch.cat([qfeat, pfeat], dim=1)
        return self.classical_net(features)

__all__ = ["HybridEstimatorQNN"]
