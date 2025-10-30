"""Quantum sampler with multi‑layer variational circuit.

The class :class:`SamplerQNN` builds a parameterised circuit that
supports:
* Multiple entangling layers.
* Input and weight parameters.
* State‑vector sampling via a simulator.
* Gradient computation with the parameter‑shift rule.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import Sampler as BaseSampler


class SamplerQNN:
    """Variational quantum sampler.

    Parameters
    ----------
    num_qubits : int, default 2
        Number of qubits in the circuit.
    num_layers : int, default 3
        Number of entangling layers.
    backend : Backend, optional
        Quantum backend for state‑vector simulation. Defaults to Aer.get_backend('statevector_simulator').
    """

    def __init__(
        self,
        num_qubits: int = 2,
        num_layers: int = 3,
        backend: Backend | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.backend = backend or Aer.get_backend("statevector_simulator")
        self.circuit = self._build_circuit()
        self.sampler = BaseSampler(backend=self.backend)
        self.sampler_qnn = QiskitSamplerQNN(
            circuit=self.circuit,
            input_params=self.circuit.parameters[:num_qubits],
            weight_params=self.circuit.parameters[num_qubits:],
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a deep variational circuit."""
        qc = QuantumCircuit(self.num_qubits)
        # Input rotations
        inputs = ParameterVector("input", self.num_qubits)
        for i in range(self.num_qubits):
            qc.ry(inputs[i], i)

        # Entangling layers
        weights = ParameterVector("weight", self.num_qubits * self.num_layers)
        w_iter = iter(weights)
        for _ in range(self.num_layers):
            # Ry rotations with trainable weights
            for i in range(self.num_qubits):
                qc.ry(next(w_iter), i)
            # CX chain
            for i in range(self.num_qubits - 1):
                qc.cx(i, i + 1)
            # Additional Ry rotations
            for i in range(self.num_qubits):
                qc.ry(next(w_iter), i)

        return qc

    def get_circuit(self) -> QuantumCircuit:
        """Return the underlying quantum circuit."""
        return self.circuit

    def sample(
        self,
        input_vals: np.ndarray,
        weight_vals: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """Execute the sampler and return measurement probabilities.

        Parameters
        ----------
        input_vals : np.ndarray
            Array of shape (num_qubits,) with input rotation angles.
        weight_vals : np.ndarray
            Array of shape (num_qubits * num_layers * 2,) with weight angles.
        shots : int
            Number of measurement shots.

        Returns
        -------
        np.ndarray
            Probability distribution over measurement outcomes.
        """
        if input_vals.shape!= (self.num_qubits,):
            raise ValueError("input_vals must match num_qubits")
        if weight_vals.shape!= (self.num_qubits * self.num_layers * 2,):
            raise ValueError("weight_vals shape mismatch")

        param_dict = {
            **{f"input_{i}": v for i, v in enumerate(input_vals)},
            **{f"weight_{i}": v for i, v in enumerate(weight_vals)},
        }
        result = self.sampler.run(
            self.circuit,
            parameter_binds=[param_dict],
            shots=shots,
        )
        probs = result.get_counts()[0]
        # Convert to probability array
        probs_arr = np.zeros(2 ** self.num_qubits)
        for bitstring, count in probs.items():
            idx = int(bitstring, 2)
            probs_arr[idx] = count / shots
        return probs_arr

    def gradient(
        self,
        input_vals: np.ndarray,
        weight_vals: np.ndarray,
        loss_fn,
        epsilon: float = 1e-5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numerical gradient of a scalar loss with respect to inputs and weights.

        Parameters
        ----------
        input_vals : np.ndarray
            Current input angles.
        weight_vals : np.ndarray
            Current weight angles.
        loss_fn : Callable[[np.ndarray], float]
            Function that maps probability distribution to a scalar loss.
        epsilon : float
            Perturbation magnitude for finite‑difference.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Gradients w.r.t. inputs and weights.
        """
        grad_inputs = np.zeros_like(input_vals)
        grad_weights = np.zeros_like(weight_vals)

        # Input gradients
        for i in range(len(input_vals)):
            delta = np.zeros_like(input_vals)
            delta[i] = epsilon
            probs_plus = self.sample(input_vals + delta, weight_vals)
            probs_minus = self.sample(input_vals - delta, weight_vals)
            grad_inputs[i] = (loss_fn(probs_plus) - loss_fn(probs_minus)) / (2 * epsilon)

        # Weight gradients
        for i in range(len(weight_vals)):
            delta = np.zeros_like(weight_vals)
            delta[i] = epsilon
            probs_plus = self.sample(input_vals, weight_vals + delta)
            probs_minus = self.sample(input_vals, weight_vals - delta)
            grad_weights[i] = (loss_fn(probs_plus) - loss_fn(probs_minus)) / (2 * epsilon)

        return grad_inputs, grad_weights


__all__ = ["SamplerQNN"]
