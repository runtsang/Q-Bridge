import torch
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals
import torchquantum as tq
import torch.nn as nn

class QuantumAutoencoder:
    """
    Variational circuit that transforms a classical latent vector into a
    quantum‑processed latent representation.  The circuit is built using
    RealAmplitudes and is evaluated with a Qiskit sampler.
    """

    def __init__(
        self,
        latent_dim: int,
        num_qubits: int,
        reps: int = 5,
        seed: int = 42,
    ) -> None:
        algorithm_globals.random_seed = seed
        self.latent_dim = latent_dim
        self.num_qubits = num_qubits
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=Sampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Build a RealAmplitudes ansatz that will be variationally trained."""
        ansatz = RealAmplitudes(self.num_qubits, reps=self.reps)
        return ansatz.to_circuit()

    def evaluate(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Run the quantum circuit for a batch of latent vectors.

        Parameters
        ----------
        latent_vector : torch.Tensor
            Tensor of shape (batch, latent_dim) to be encoded into the
            circuit parameters.  The encoding is done by mapping each element
            to a rotation angle on a separate qubit.
        """
        batch = latent_vector.shape[0]
        params = latent_vector.to(torch.float64).numpy()
        # Ensure the parameters array matches the circuit parameter count
        total_params = len(self.circuit.parameters)
        if params.shape[1]!= total_params:
            # trim or pad the parameters
            if params.shape[1] > total_params:
                params = params[:, :total_params]
            else:
                pad = np.zeros((batch, total_params - params.shape[1]), dtype=np.float64)
                params = np.concatenate([params, pad], axis=1)
        outputs = self.qnn.forward(params)
        return torch.from_numpy(outputs).float()

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """
    Random 2‑qubit quantum kernel applied to 2×2 image patches.
    This is a direct quantum analogue of the classical quanvolution filter.
    """

    def __init__(self, n_wires: int = 4, patch_size: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.patch_size = patch_size
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum kernel to all non‑overlapping 2×2 patches of the
        input image.

        Parameters
        ----------
        x : torch.Tensor
            Batch of images of shape (batch, 1, 28, 28) or (batch, 28, 28).
        """
        bsz = x.shape[0]
        device = x.device
        if x.ndim == 4:
            x = x.squeeze(1)
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, self.n_wires))
        return torch.cat(patches, dim=1)

__all__ = ["QuantumAutoencoder", "QuantumQuanvolutionFilter"]
