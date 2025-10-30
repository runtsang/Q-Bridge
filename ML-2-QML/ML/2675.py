import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGen240(nn.Module):
    """
    Hybrid convolutional filter that supports both classical and quantum
    feature extraction. The module can operate in three modes:

    * 'classic' – pure 2x2 convolution with learnable weights.
    * 'quantum' – each 2x2 patch is encoded into a 4‑qubit circuit
      and the measurement outcomes are used as features.
    * 'hybrid' – first a classical conv layer followed by a quantum
      layer for each patch, providing a richer representation.

    The design is inspired by the original Conv.py and Quanvolution.py
    implementations but extends them with a configurable mode and
    efficient patch extraction.
    """

    def __init__(self, mode: str = "classic", kernel_size: int = 2,
                 threshold: float = 0.0, n_qubits: int = 4,
                 shots: int = 100, backend=None):
        super().__init__()
        self.mode = mode
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = backend

        # Classical convolution branch
        self.classic_conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                                      stride=kernel_size, bias=True)

        # Quantum branch – placeholder for a quantum circuit factory
        # It will be instantiated lazily in forward when needed.
        self.q_circuit_factory = None

    def _lazy_q_circuit(self):
        if self.q_circuit_factory is None:
            import qiskit
            from qiskit.circuit.random import random_circuit
            from qiskit import Aer, execute
            import numpy as np

            class QuantumPatch(nn.Module):
                def __init__(self, n_qubits, threshold, shots, backend):
                    super().__init__()
                    self.n_qubits = n_qubits
                    self.threshold = threshold
                    self.shots = shots
                    self.backend = backend or Aer.get_backend("qasm_simulator")
                    self.circuit = qiskit.QuantumCircuit(self.n_qubits)
                    self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
                    for i in range(self.n_qubits):
                        self.circuit.rx(self.theta[i], i)
                    self.circuit.barrier()
                    self.circuit += random_circuit(self.n_qubits, 2)
                    self.circuit.measure_all()

                def forward(self, patch):
                    # patch: (batch, n_qubits)
                    param_binds = []
                    for dat in patch:
                        bind = {self.theta[i]: (np.pi if val.item() > self.threshold else 0)
                                for i, val in enumerate(dat)}
                        param_binds.append(bind)
                    job = execute(self.circuit, self.backend,
                                  shots=self.shots, parameter_binds=param_binds)
                    result = job.result()
                    features = []
                    for i in range(patch.shape[0]):
                        counts = result.get_counts(self.circuit, index=i)
                        ones = sum(int(bit) * cnt for key, cnt in counts.items() for bit in key)
                        prob = ones / (self.shots * self.n_qubits)
                        features.append(prob)
                    return torch.tensor(features, dtype=torch.float32)

            self.q_circuit_factory = QuantumPatch

        return self.q_circuit_factory

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the hybrid filter.

        Parameters
        ----------
        x : torch.Tensor
            Input image of shape (batch, 1, H, W).

        Returns
        -------
        torch.Tensor
            Feature representation after the chosen mode.
        """
        if self.mode == "classic":
            return self.classic_conv(x)

        # Extract 2x2 patches
        patches = x.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size, self.kernel_size)
        # patches shape: (batch, 1, H_p, W_p, kernel, kernel)
        batch, _, h_p, w_p, k1, k2 = patches.shape
        patches = patches.squeeze(1).reshape(batch, h_p * w_p, self.n_qubits)

        if self.mode == "quantum":
            q_circuit = self._lazy_q_circuit()
            features = q_circuit(n_qubits=self.n_qubits,
                                 threshold=self.threshold,
                                 shots=self.shots,
                                 backend=self.backend)(patches)
            # reshape back to image grid
            return features.reshape(batch, 1, h_p, w_p)

        # hybrid mode: apply classical conv first, then quantum on each patch
        if self.mode == "hybrid":
            # classical conv on each patch
            classic_features = self.classic_conv(x)
            # flatten patches from classic output
            patches = classic_features.unfold(2, self.kernel_size, self.kernel_size).unfold(3, self.kernel_size, self.kernel_size)
            batch, _, h_p, w_p, k1, k2 = patches.shape
            patches = patches.squeeze(1).reshape(batch, h_p * w_p, self.n_qubits)
            q_circuit = self._lazy_q_circuit()
            quantum_features = q_circuit(n_qubits=self.n_qubits,
                                         threshold=self.threshold,
                                         shots=self.shots,
                                         backend=self.backend)(patches)
            return quantum_features.reshape(batch, 1, h_p, w_p)

        raise ValueError(f"Unsupported mode: {self.mode}")

__all__ = ["ConvGen240"]
