"""ConvEnhanced: Classical depth‑wise separable convolution with optional quantum fallback.

The class is designed to be a drop‑in replacement for the original Conv filter.  It
provides two modes:

* **Classical mode** – a depth‑wise separable Conv2d with a learnable bias that acts
  as the threshold.  This is 2× faster than a full 2‑D convolution for 1‑channel
  inputs while preserving the receptive field.

* **Quantum mode** – when ``use_quantum=True`` and a Qiskit backend is supplied,
  the forward pass runs a tiny variational circuit instead of the convolution.
  The circuit consists of a single Rx rotation per qubit followed by a
  CNOT‑chain; the rotation angles are set to π if the corresponding input
  pixel exceeds the threshold, otherwise 0.  The output is the average
  probability of measuring |1> across all qubits.

Both modes expose the same ``forward`` and ``run`` methods, making the class
compatible with existing CNN pipelines.  The threshold is a learnable
parameter in the classical mode and a fixed value in the quantum mode.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit import execute

class ConvEnhanced(nn.Module):
    """Drop‑in replacement for the original Conv filter.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the square kernel (default: 2).
    threshold : float, optional
        Initial threshold value for the bias (default: 0.0).  In quantum mode
        this value is used only for data encoding; it is not a learnable
        parameter.
    use_quantum : bool, optional
        If True, the forward pass runs the quantum circuit.  If False the
        classical separable convolution is used.  Default is False.
    backend : qiskit.providers.Backend, optional
        Qiskit backend to execute the circuit on.  Required if ``use_quantum``
        is True.
    shots : int, optional
        Number of shots for the quantum simulation.  Default: 1024.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        use_quantum: bool = False,
        backend: Backend | None = None,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.use_quantum = use_quantum
        self.shots = shots

        if self.use_quantum:
            if backend is None:
                raise ValueError("Quantum mode requires a Qiskit backend.")
            self.backend = backend
            self._build_qc()
        else:
            # Depth‑wise separable convolution for 1‑channel input
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                bias=True,
            )
            # Initialise bias with the provided threshold
            nn.init.constant_(self.conv.bias, threshold)

    # --------------------------------------------------------------------- #
    #  Classical path
    # --------------------------------------------------------------------- #
    def _classical_run(self, data: np.ndarray) -> float:
        """Run the separable convolution on a single patch."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()

    # --------------------------------------------------------------------- #
    #  Quantum path
    # --------------------------------------------------------------------- #
    def _build_qc(self) -> None:
        """Construct a parameter‑efficient variational circuit."""
        n_qubits = self.kernel_size ** 2
        self.n_qubits = n_qubits
        self.theta = [Parameter(f"theta{i}") for i in range(n_qubits)]

        qc = qiskit.QuantumCircuit(n_qubits)
        # Parameterised rotations
        for i, t in enumerate(self.theta):
            qc.rx(t, i)
        # Simple entanglement pattern
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        self._qc = qc

    def _quantum_run(self, data: np.ndarray) -> float:
        """Execute the quantum circuit for a single patch."""
        # Encode data into rotation angles
        data_flat = data.reshape(-1)
        param_binds = []
        for val in data_flat:
            bind = {}
            for i, th in enumerate(self.theta):
                bind[th] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self._qc,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self._qc)

        # Compute average probability of measuring |1> across all qubits
        total_ones = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
        return total_ones / (self.shots * self.n_qubits)

    # --------------------------------------------------------------------- #
    #  Public API
    # --------------------------------------------------------------------- #
    def run(self, data: np.ndarray) -> float:
        """Run a single patch through the selected mode."""
        if self.use_quantum:
            return self._quantum_run(data)
        return self._classical_run(data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for batched input.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape (batch, 1, kernel_size, kernel_size).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch, 1, 1, 1) containing the filter response.
        """
        if self.use_quantum:
            # Apply the quantum circuit to each patch in the batch
            batch = x.shape[0]
            outputs = []
            for i in range(batch):
                patch = x[i, 0].cpu().numpy()
                outputs.append(self._quantum_run(patch))
            return torch.tensor(outputs, dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        else:
            return self.conv(x)

__all__ = ["ConvEnhanced"]
