"""ConvEnhanced: quantum‑enhanced convolutional filter.

This module implements a parameter‑shared variational circuit that
acts as a quantum filter.  It can be trained with gradient‑based
optimisers via the parameter‑shift rule and can be mixed with a
classical convolution for hybrid inference.

Key features
------------
* Parameter‑shared ansatz across qubits for scalability.
* Entangling layers with CNOT rings.
* Continuous mapping of pixel values to rotation angles.
* Hybrid evaluation that blends classical and quantum outputs.
* Exposes a `run` method for inference and a `trainable_params`
  property for optimisation.

"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes

class ConvEnhanced:
    """
    Quantum convolutional filter implemented as a variational circuit.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the filter (kernel_size × kernel_size).
    backend : qiskit.providers.Backend, optional
        Backend to execute the circuit.  Defaults to Aer qasm_simulator.
    shots : int, default 1024
        Number of shots per execution.
    threshold : float, default 0.5
        Pixel value threshold for mapping to rotation angles.
    n_layers : int, default 2
        Number of entangling layers in the ansatz.
    param_sharing : bool, default True
        If True, use a single parameter per layer shared across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 1024,
        threshold: float = 0.5,
        n_layers: int = 2,
        param_sharing: bool = True,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.n_layers = n_layers
        self.param_sharing = param_sharing

        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Build parameter‑shared ansatz
        if self.param_sharing:
            self.params = [Parameter(f"theta_{l}") for l in range(self.n_layers)]
        else:
            self.params = [
                Parameter(f"theta_{l}_{q}") for l in range(self.n_layers) for q in range(self.n_qubits)
            ]

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the variational circuit."""
        qc = QuantumCircuit(self.n_qubits)

        # Initial single‑qubit rotations
        for q in range(self.n_qubits):
            idx = 0 if self.param_sharing else q
            qc.ry(self.params[idx], q)

        # Entangling layers
        for l in range(1, self.n_layers):
            # CNOT ring
            for q in range(self.n_qubits):
                qc.cx(q, (q + 1) % self.n_qubits)
            # Parameterized rotations
            for q in range(self.n_qubits):
                idx = l if self.param_sharing else l * self.n_qubits + q
                qc.ry(self.params[idx], q)

        qc.measure_all()
        return qc

    def _data_to_params(self, data: np.ndarray) -> Dict[Parameter, float]:
        """
        Map pixel values to rotation angles.

        Parameters
        ----------
        data : np.ndarray
            2D array of shape (kernel_size, kernel_size).

        Returns
        -------
        dict
            Mapping from circuit parameters to float values.
        """
        flat = data.reshape(-1)
        # Scale pixel intensity to [0, π] based on threshold
        angles = np.pi * (flat > self.threshold).astype(float)
        if self.param_sharing:
            # Average angle across all qubits for each layer
            angle_per_layer = [np.mean(angles) for _ in range(self.n_layers)]
            return {p: angle for p, angle in zip(self.params, angle_per_layer)}
        else:
            return {p: angle for p, angle in zip(self.params, angles)}

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single data patch.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Normalised expectation value of Z on all qubits (mean probability of |1>).
        """
        bind = self._data_to_params(data)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average number of |1> outcomes
        total_ones = 0
        for bitstring, freq in counts.items():
            total_ones += freq * bitstring.count("1")
        prob_one = total_ones / (self.shots * self.n_qubits)
        return prob_one

    def hybrid_run(self, data: np.ndarray, classical_filter: "ConvEnhanced") -> float:
        """
        Hybrid inference that mixes classical and quantum outputs.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size).
        classical_filter : ConvEnhanced
            Instance of the classical ConvEnhanced filter.

        Returns
        -------
        float
            Weighted sum of classical mean activation and quantum probability.
        """
        # Classical mean activation
        import torch
        tensor = torch.as_tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        class_act = classical_filter.mean_activation(tensor).item()

        # Quantum probability
        q_prob = self.run(data)

        # Simple weighted average (weights can be tuned)
        return 0.5 * class_act + 0.5 * q_prob

    @property
    def trainable_params(self):
        """
        Return a list of trainable parameters for optimisation.

        Returns
        -------
        list
            List of qiskit.circuit.Parameter objects.
        """
        return self.params

    @classmethod
    def from_dict(cls, cfg: Dict) -> "ConvEnhanced":
        """
        Construct the quantum filter from a dictionary of hyper‑parameters.

        Parameters
        ----------
        cfg : dict
            Dictionary containing any of the constructor arguments.

        Returns
        -------
        ConvEnhanced
            Instantiated quantum filter.
        """
        return cls(**cfg)

__all__ = ["ConvEnhanced"]
