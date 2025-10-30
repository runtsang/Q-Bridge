"""Quantum sampler network with a configurable variational circuit.

The implementation uses PennyLane and a default qiskit statevector simulator.
"""

import pennylane as qml
import pennylane.numpy as pnp
from pennylane import Device
from typing import Sequence

class _QuantumSampler:
    """Variational quantum sampler.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    layers : int, default 2
        Number of variational layers.
    device : Device | None, default None
        PennyLane quantum device. If ``None`` a default qiskit statevector simulator
        is used.
    """

    def __init__(self, num_qubits: int = 2, layers: int = 2, device: Device | None = None) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = device or qml.device("qiskit.aer.statevector_simulator", wires=num_qubits)
        self.weights = pnp.random.uniform(0, 2 * pnp.pi, (layers, num_qubits))
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> qml.QNode:
        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: pnp.ndarray, weights: pnp.ndarray) -> pnp.ndarray:
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RY(weights[l, i], wires=i)
                # entangle all neighbours in a ring
                for i in range(self.num_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.num_qubits])
            return qml.probs(wires=range(self.num_qubits))
        return circuit

    def sample(self, inputs: torch.Tensor, n_shots: int = 1024) -> torch.Tensor:
        """Return a probability distribution estimated by measurement.

        Parameters
        ----------
        inputs : torch.Tensor
            Shape ``(batch_size, num_qubits)``.
        n_shots : int, default 1024
            Number of shots for sampling.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, 2**num_qubits)`` containing empirical probabilities.
        """
        probs = []
        for inp in inputs:
            probs.append(self.circuit(inp.detach().numpy(), self.weights))
        probs = pnp.stack(probs)
        return torch.tensor(probs, dtype=torch.float32)

def SamplerQNN(**kwargs) -> _QuantumSampler:
    """Convenience factory returning a SamplerQNN instance.

    Example
    -------
    >>> q_model = SamplerQNN(num_qubits=3, layers=3)
    """
    return _QuantumSampler(**kwargs)

__all__ = ["SamplerQNN"]
