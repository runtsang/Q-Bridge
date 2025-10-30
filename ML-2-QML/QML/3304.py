import torch
import torch.nn as nn
import torch.nn.functional as F
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator

class QuantumQuanvolutionFilter(nn.Module):
    """
    Quantum patch‑wise feature extractor inspired by the Quanvolution example.
    Each 2×2 image patch is encoded into a 4‑qubit circuit via Ry rotations,
    followed by a variational layer and measurement of a Pauli‑Z observable.
    """
    def __init__(self, patch_size=2, n_wires=4):
        super().__init__()
        self.patch_size = patch_size
        self.n_wires = n_wires
        # Input and trainable weight parameters
        self.input_params = [Parameter(f"input_{i}") for i in range(n_wires)]
        self.weight_params = [Parameter(f"weight_{i}") for i in range(n_wires)]

        # Build the variational circuit
        self.circuit = QuantumCircuit(n_wires)
        for i in range(n_wires):
            self.circuit.ry(self.input_params[i], i)
            self.circuit.rx(self.weight_params[i], i)
        # Random rotation layer (fixed for all patches)
        for i in range(n_wires):
            self.circuit.rz(Parameter(f"rz_{i}"), i)
        self.circuit.measure_all()

        # Observable for expectation value
        self.observable = SparsePauliOp.from_list([("Z"*n_wires, 1)])
        self.estimator = StatevectorEstimator()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[self.observable],
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=self.estimator,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract quantum features for each 2×2 patch in the image.
        Returns a tensor of shape (B, num_patches).
        """
        bsz, _, h, w = x.shape
        patches_per_dim = h // self.patch_size
        features = []

        for r in range(0, h, self.patch_size):
            for c in range(0, w, self.patch_size):
                # Flatten patch to vector of length n_wires
                patch = x[:, :, r:r+self.patch_size, c:c+self.patch_size]
                flat = patch.view(bsz, -1)  # shape: (B, n_wires)
                # Prepare parameter dictionary for each sample in the batch
                batch_vals = torch.empty(bsz, 1)
                for i in range(bsz):
                    param_dict = {p: float(v) for p, v in zip(self.input_params, flat[i].tolist())}
                    # weight_params are trainable; we let EstimatorQNN handle them
                    param_dict.update({w: 0.0 for w in self.weight_params})
                    val = self.estimator_qnn.predict(param_dict)[0]
                    batch_vals[i] = torch.tensor(val)
                features.append(batch_vals)

        # Concatenate features from all patches
        return torch.cat(features, dim=1)

class HybridEstimatorQNN(nn.Module):
    """
    Hybrid quantum‑classical regression network.
    The quantum quanvolution filter produces a feature map that is linearly
    combined to output the target value.  Training optimizes both the
    variational weights and the linear head simultaneously.
    """
    def __init__(self, patch_size=2, n_wires=4, output_dim=1):
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter(patch_size, n_wires)
        num_patches = (28 // patch_size) ** 2
        self.linear = nn.Linear(n_wires * num_patches, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)            # shape: (B, n_wires * num_patches)
        out = self.linear(features)           # shape: (B, output_dim)
        return out

__all__ = ["HybridEstimatorQNN"]
