import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
from qiskit import Aer, execute
import torch
import torch.nn as nn

class ConvGen240(nn.Module):
    """
    Quantum‑centric implementation of the hybrid filter.  The class
    exposes the same API as the classical counterpart but all
    feature extraction is performed on a quantum backend.

    The filter operates on 2x2 patches: each patch is encoded into
    a 4‑qubit circuit, a random two‑qubit layer is applied, and the
    expectation value of Pauli‑Z on all qubits is returned as a
    feature.  The resulting feature map has the same spatial
    dimensions as the input image.

    Parameters
    ----------
    kernel_size : int
        Size of the square patch (default 2).
    shots : int
        Number of shots used in the simulation.
    threshold : float
        Threshold for encoding classical values into rotation angles.
    backend : qiskit.providers.BaseBackend, optional
        Quantum backend to execute the circuits.  If None, the
        Aer qasm simulator is used.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 100,
                 threshold: float = 0.0, backend=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Pre‑build a reusable circuit template
        self.base_circuit = qiskit.QuantumCircuit(self.kernel_size**2)
        self.params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.kernel_size**2)]
        for i, p in enumerate(self.params):
            self.base_circuit.rx(p, i)
        self.base_circuit.barrier()
        self.base_circuit += random_circuit(self.kernel_size**2, 2)
        self.base_circuit.measure_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that runs the quantum circuit on each 2x2 patch.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Feature map of shape (batch, 1, H//kernel_size, W//kernel_size).
        """
        batch, _, H, W = x.shape
        # Extract patches
        patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size, self.kernel_size)
        # patches shape: (batch, 1, H_p, W_p, k, k)
        patches = patches.squeeze(1).reshape(batch, -1, self.kernel_size**2)

        # Prepare parameter bindings
        param_binds = []
        for patch in patches:
            bind = {p: (np.pi if val.item() > self.threshold else 0)
                    for p, val in zip(self.params, patch)}
            param_binds.append(bind)

        # Execute the circuit on all patches in a batch
        job = execute(self.base_circuit, self.backend,
                      shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = np.zeros(patches.shape[0], dtype=np.float32)
        for idx in range(patches.shape[0]):
            counts[idx] = sum(int(bit) * val for key, val in result.get_counts(self.base_circuit, index=idx).items() for bit in key) / (self.shots * self.kernel_size**2)

        # Reshape back to image grid
        H_p = H // self.kernel_size
        W_p = W // self.kernel_size
        return torch.from_numpy(counts.reshape(batch, 1, H_p, W_p))

__all__ = ["ConvGen240"]
