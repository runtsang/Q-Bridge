import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class ConvFraudHybridQML:
    """
    Hybrid quantum‑classical model that replaces the classical convolution with a
    variational Qiskit circuit.  The circuit is executed on the supplied backend
    and the resulting probability of measuring |1> is fed into a classical
    fraud‑detection network.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 127,
        fraud_input: FraudLayerParameters | None = None,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
        backend=None,
        shots: int = 100,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")

        n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += random_circuit(n_qubits, 2)
        self._circuit.measure_all()

        if fraud_input is None:
            fraud_input = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layers is None:
            fraud_layers = []

        self.fraud_network = build_fraud_detection_program(fraud_input, fraud_layers)

    def run(self, data: np.ndarray) -> torch.Tensor:
        """
        Execute the quantum circuit on the provided 2‑D data array, convert the
        measurement statistics into a scalar, and forward it through the
        fraud‑detection network.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Fraud‑risk score of shape (1,).
        """
        data = np.reshape(data, (1, self.kernel_size ** 2))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(b) for b in key)
            counts += ones * val
        prob = counts / (self.shots * self.kernel_size ** 2)

        # Fraud network expects a 2‑D vector; duplicate the probability
        fraud_input = torch.tensor([prob, prob], dtype=torch.float32).unsqueeze(0)
        out = self.fraud_network(fraud_input)
        return out.squeeze()

__all__ = ["ConvFraudHybridQML", "FraudLayerParameters", "build_fraud_detection_program"]
