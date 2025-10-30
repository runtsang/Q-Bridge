"""Hybrid convolutional filter with optional quantum integration.

The original `Conv` seed defined a simple 2‑D convolution whose output
is a single scalar.  This extension keeps the same interface while
adding support for multiple kernel sizes, a learnable threshold
and an optional quantum layer for uncertainty estimation.

The class `ConvHybrid` inherits from `torch.nn.Module` so it can be
instantiated inside any CNN.  The `run` method accepts a 2‑D NumPy
array of shape `(kernel_size, kernel_size)` and returns a `float`.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np

__all__ = ["ConvHybrid"]

class ConvHybrid(nn.Module):
    """Hybrid convolutional filter that mixes classical and quantum
    processing.  The module can be configured with several kernel
    sizes; each kernel produces a scalar that is fused with learnable
    weights.  Optionally, a quantum variational circuit can be used
    as part of the fusion.

    Parameters
    ----------
    kernel_sizes : list[int], optional
        List of kernel sizes to apply.  If ``None`` defaults to ``[2]``.
    threshold : float, optional
        Threshold used by the sigmoid activation (classical) and by
        the data-to‑parameter mapping (quantum).  If ``None`` defaults
        to ``0.0`` for classical and ``127`` for quantum.
    n_quantum_layers : int, optional
        Number of variational quantum layers to append after the
        classical conv outputs.  If ``0`` the layer is purely
        classical.  The quantum part is implemented with Qiskit and
        executed on the Aer simulator.
    """

    def __init__(
        self,
        kernel_sizes: list[int] | None = None,
        threshold: float | None = None,
        n_quantum_layers: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kernel_sizes = kernel_sizes or [2]
        self.threshold = threshold if threshold is not None else 0.0
        self.n_quantum_layers = n_quantum_layers

        # Create a convolution for each kernel size
        self.convs = nn.ModuleList()
        for k in self.kernel_sizes:
            conv = nn.Conv2d(1, 1, kernel_size=k, bias=True)
            self.convs.append(conv)

        # Fusion weights for each kernel
        self.fusion_weights = nn.Parameter(
            torch.ones(len(self.kernel_sizes))
        )

        # Optional quantum backend
        self.use_quantum = self.n_quantum_layers > 0
        if self.use_quantum:
            import qiskit
            from qiskit.circuit import Parameter
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = 1024

            # Use the first kernel size for the quantum circuit
            k = self.kernel_sizes[0]
            self.n_qubits = k ** 2
            self.circuit_template = qiskit.QuantumCircuit(self.n_qubits)
            self.theta = [Parameter(f"theta_{i}") for i in range(self.n_qubits)]
            for i in range(self.n_qubits):
                self.circuit_template.ry(self.theta[i], i)
            # Simple entanglement pattern
            for i in range(0, self.n_qubits - 1, 2):
                self.circuit_template.cx(i, i + 1)
                self.circuit_template.cx(i + 1, i)
            self.circuit_template.measure_all()

    def _run_quantum(self, data: np.ndarray) -> float:
        """Run a variational circuit on the given data and return the
        average probability of measuring |1> across all qubits."""
        import qiskit
        from qiskit import execute

        bind = {}
        for i, val in enumerate(data.flatten()):
            bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
        param_binds = [bind]

        job = execute(
            self.circuit_template,
            backend=self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit_template)

        total_ones = 0
        total_counts = 0
        for bitstring, freq in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * freq
            total_counts += freq
        return total_ones / (total_counts * self.n_qubits)

    def run(self, data: np.ndarray) -> float:
        """Compute the hybrid filter output for a 2‑D array.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape ``(kernel_size, kernel_size)``.
        Returns
        -------
        float
            Scalar output of the filter.
        """
        results = []
        for conv in self.convs:
            # Prepare tensor
            tensor = torch.as_tensor(data, dtype=torch.float32)
            # Reshape to match conv input: (batch, channel, H, W)
            tensor = tensor.view(1, 1, data.shape[0], data.shape[1])
            logits = conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            results.append(activations.mean().item())

        # Fuse using learnable weights
        fused = sum(w * r for w, r in zip(self.fusion_weights, results))

        # Optionally add quantum contribution
        if self.use_quantum:
            q_val = self._run_quantum(data)
            fused += q_val

        return fused
