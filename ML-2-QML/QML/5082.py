"""Quantum‑centric autoencoder.

The :class:`AutoencoderGen` defined in this module implements a
variational quantum autoencoder that can be trained on a local simulator
or a Braket/IBMQ backend.  It draws inspiration from the Autoencoder
QML seed, the EstimatorQNN example, the quantum convolution filter,
and the fully‑connected quantum layer.  The quantum part is a
parameterised RealAmplitudes circuit that refines a classical latent
vector.  A classical fully‑connected decoder reconstructs the input.

Features
--------
- Variational quantum encoder with swap‑test style measurement.
- Optional quantum convolution pre‑processing of 2‑D data.
- Classical decoder using a small FC network.
- Training loop that optimises the decoder parameters; quantum
  parameters can be updated with Qiskit optimisers.
"""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.random import random_circuit
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, Estimator
from qiskit import Aer
from qiskit.algorithms.optimizers import COBYLA


class QConvFilter:
    """Quantum convolution filter that mimics the Conv example."""

    def __init__(self, kernel_size: int, backend: str = "qasm_simulator",
                 shots: int = 512, threshold: float = 0.5):
        self.n_qubits = kernel_size ** 2
        self.kernel_size = kernel_size
        self.backend = Aer.get_backend(backend)
        self.shots = shots
        self.threshold = threshold

        # Build the circuit
        self.qr = QuantumRegister(self.n_qubits, "q")
        self.cr = ClassicalRegister(self.n_qubits, "c")
        self.circuit = QuantumCircuit(self.qr, self.cr)

        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], self.qr[i])

        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure(self.qr, self.cr)

    def run(self, patch: np.ndarray) -> float:
        """Run the filter on a 2‑D patch and return the average |1> probability."""
        if patch.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError(f"Expected patch of shape {(self.kernel_size, self.kernel_size)}")

        flat = patch.reshape(1, self.n_qubits)
        param_binds = []
        for row in flat:
            bind = {theta: np.pi if val > self.threshold else 0.0
                    for theta, val in zip(self.theta, row)}
            param_binds.append(bind)

        job = qiskit.execute(self.circuit, self.backend, shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self.circuit)

        total = 0.0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)


class AutoencoderGen(nn.Module):
    """Quantum‑centric autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector (assumed to be a square number).
    latent_dim : int, default 32
        Size of the latent space.
    quantum_reps : int, default 5
        Number of repetitions for the RealAmplitudes ansatz.
    quantum_backend : str, default "qasm_simulator"
        Backend used for the quantum part (e.g. "qasm_simulator",
        "statevector_simulator", or an IBMQ provider).
    quantum_shots : int, default 1024
        Shots for the quantum sampler.
    conv_kernel : int, optional
        Size of a 2‑D quantum convolution filter applied to the data.
    conv_threshold : float, default 0.5
        Threshold used by the convolution filter.
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 quantum_reps: int = 5,
                 quantum_backend: str = "qasm_simulator",
                 quantum_shots: int = 1024,
                 conv_kernel: int | None = None,
                 conv_threshold: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.quantum_reps = quantum_reps
        self.quantum_backend = quantum_backend
        self.quantum_shots = quantum_shots
        self.conv_kernel = conv_kernel
        self.conv_threshold = conv_threshold

        # Quantum convolution pre‑processing if requested
        if self.conv_kernel is not None:
            self.qconv = QConvFilter(kernel_size=self.conv_kernel,
                                     backend=self.quantum_backend,
                                     shots=self.quantum_shots,
                                     threshold=self.conv_threshold)
            size = int(np.sqrt(self.input_dim))
            if size % self.conv_kernel!= 0:
                raise ValueError("input_dim must be a perfect square divisible by conv_kernel")
            patches_per_dim = size // self.conv_kernel
            self.preprocess_dim = patches_per_dim ** 2
        else:
            self.preprocess_dim = self.input_dim

        # Classical decoder network
        dec_layers = []
        in_dim = self.latent_dim
        hidden = 128
        dec_layers.append(nn.Linear(in_dim, hidden))
        dec_layers.append(nn.ReLU())
        dec_layers.append(nn.Linear(hidden, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Build the quantum encoder circuit
        self._build_quantum_encoder()

        # Linear layer that maps pre‑processed data to an initial latent
        self.initial_latent_layer = nn.Linear(self.preprocess_dim, self.latent_dim)

    def _build_quantum_encoder(self):
        """Construct a SamplerQNN that maps a classical latent vector to a refined latent."""
        qr = QuantumRegister(self.latent_dim, "q")
        qc = QuantumCircuit(qr)

        self.input_params = [Parameter(f"x{i}") for i in range(self.latent_dim)]
        for i, theta in enumerate(self.input_params):
            qc.rx(theta, qr[i])

        ansatz = RealAmplitudes(self.latent_dim, reps=self.quantum_reps)
        qc += ansatz(qr)

        self.weight_params = qc.parameters
        self.quantum_sampler = StatevectorSampler()
        self.quantum_encoder = SamplerQNN(circuit=qc,
                                          input_params=self.input_params,
                                          weight_params=self.weight_params,
                                          output_shape=self.latent_dim,
                                          sampler=self.quantum_sampler)

    def _preprocess(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the quantum convolution filter to 2‑D data patches."""
        if self.conv_kernel is None:
            return inputs
        batch, dim = inputs.shape
        size = int(np.sqrt(dim))
        patches_per_dim = size // self.conv_kernel
        out = []
        for b in range(batch):
            patch_vector = inputs[b].cpu().numpy().reshape(size, size)
            features = []
            for i in range(patches_per_dim):
                for j in range(patches_per_dim):
                    patch = patch_vector[i*self.conv_kernel:(i+1)*self.conv_kernel,
                                         j*self.conv_kernel:(j+1)*self.conv_kernel]
                    val = self.qconv.run(patch)
                    features.append(val)
            out.append(features)
        return torch.tensor(out, dtype=inputs.dtype, device=inputs.device)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the input into the latent space."""
        processed = self._preprocess(inputs)
        init_latent = self.initial_latent_layer(processed)
        # Refine with quantum circuit
        qt_latent = self.quantum_encoder(init_latent.numpy(),
                                         return_type="numpy")
        return torch.tensor(qt_latent, dtype=init_latent.dtype,
                            device=init_latent.device)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back to the input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """End‑to‑end reconstruction."""
        return self.decode(self.encode(inputs))

    def train(self,
              data: torch.Tensor,
              *,
              epochs: int = 100,
              batch_size: int = 64,
              lr: float = 1e-3,
              weight_decay: float = 0.0,
              device: torch.device | None = None) -> list[float]:
        """Training loop that optimises the decoder parameters; quantum
        parameters are kept fixed for demonstration but can be optimised
        with Qiskit optimisers."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(data.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_decoder = torch.optim.Adam(self.decoder.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                opt_decoder.zero_grad()
                loss.backward()
                opt_decoder.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


__all__ = ["AutoencoderGen"]
