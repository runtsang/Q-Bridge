"""
Quantum feature extractor that underpins ``SamplerQNNGen312``.
The design merges the SamplerQNN and EstimatorQNN templates,
adds a fully‑connected quantum layer (inspired by FCL),
and is compatible with the QuantumRegression module.

The circuit is parameterised by two groups of angles:
* ``input_params`` – encode the classical features via Ry gates.
* ``weight_params`` – trainable ansatz that provides learnable
  transformations before measurement.

The module returns probability distributions over the computational
basis, which the classical head consumes.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class QuantumSampler:
    """
    Parameterised quantum circuit that maps classical inputs to
    a probability distribution over 2^n_qubits outcomes.

    Parameters
    ----------
    n_qubits : int, default 2
        Number of qubits in the circuit.
    device : str | qiskit.providers.Provider, optional
        Backend name or provider; defaults to Aer state‑vector simulator.
    """

    def __init__(self, n_qubits: int = 2, device: str | None = None) -> None:
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("statevector_simulator") if device is None else device

        # Parameter vectors
        self.input_params = ParameterVector("input", n_qubits)
        self.weight_params = ParameterVector("weight", 4)  # simple ansatz

        # Build the circuit template
        self.circuit = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.ry(self.input_params[q], q)
        self.circuit.cx(0, 1)
        for w in self.weight_params:
            self.circuit.ry(w, 0)
            self.circuit.ry(w, 1)
        self.circuit.cx(0, 1)

    def _bind_and_execute(self, params: np.ndarray) -> np.ndarray:
        """
        Bind parameters and execute the circuit to obtain the probability
        vector.

        Parameters
        ----------
        params : np.ndarray
            Shape (batch, n_qubits + 4).  The first ``n_qubits`` columns are
            the input angles; the remaining four are the trainable weights.

        Returns
        -------
        np.ndarray
            Shape (batch, 2**n_qubits) probability distributions.
        """
        batch, _ = params.shape
        probs = np.zeros((batch, 2 ** self.n_qubits), dtype=np.float64)

        for i in range(batch):
            bind_dict = dict(
                zip(
                    list(self.input_params) + list(self.weight_params),
                    params[i],
                )
            )
            bound_circ = self.circuit.bind_parameters(bind_dict)
            result = execute(bound_circ, self.backend, shots=0).result()
            statevec = result.get_statevector(bound_circ)
            probs[i] = np.abs(statevec) ** 2

        return probs

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert a batch of classical feature vectors into quantum
        probability distributions.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_features).  The first ``n_qubits`` entries are
            mapped to the input parameters; the remainder initialise the
            weight parameters.

        Returns
        -------
        torch.Tensor
            Shape (batch, 2**n_qubits) probability vectors.
        """
        import torch

        # Ensure the tensor is on CPU and converted to numpy
        arr = x.detach().cpu().numpy()
        # Pad or truncate to match expected shape
        if arr.shape[1] < self.n_qubits + 4:
            pad = np.zeros((arr.shape[0], self.n_qubits + 4 - arr.shape[1]))
            arr = np.concatenate([arr, pad], axis=1)
        elif arr.shape[1] > self.n_qubits + 4:
            arr = arr[:, : self.n_qubits + 4]
        probs = self._bind_and_execute(arr)
        return torch.from_numpy(probs).float().to(x.device)
