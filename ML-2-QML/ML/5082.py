"""Hybrid classical‑quantum autoencoder.

The :class:`AutoencoderGen` class can be used as a pure classical
autoencoder or as a hybrid model that inserts a quantum circuit
refining the latent representation.  It is inspired by the classical
Autoencoder seed, the EstimatorQNN example, the quantum fully‑connected
layer (FCL) and the quantum convolution filter (Conv) examples.
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
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler
from qiskit import Aer


class QConvFilter:
    """Quantum convolution filter that mimics the Conv example.

    The filter is a parameterised circuit that maps a square patch of
    classical data into a single scalar expectation value.  The circuit
    is executed on the Aer qasm simulator.
    """

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

        # RX gates with parameters to encode pixel values
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], self.qr[i])

        self.circuit.barrier()
        # Add a few random two‑layer gates to increase expressivity
        self.circuit += random_circuit(self.n_qubits, 2)

        # Measure all qubits
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

        # Compute probability of measuring 1 on each qubit
        total = 0.0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val
        return total / (self.shots * self.n_qubits)


class AutoencoderGen(nn.Module):
    """Hybrid autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    latent_dim : int, default 32
        Size of the latent space.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for both encoder and decoder.
    dropout : float, default 0.1
        Dropout probability.
    use_quantum : bool, default True
        If True, a quantum circuit refines the latent representation.
    quantum_reps : int, default 5
        Number of repetitions for the RealAmplitudes ansatz.
    quantum_backend : str, default "qasm_simulator"
        Qiskit backend used for the quantum part.
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
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 use_quantum: bool = True,
                 quantum_reps: int = 5,
                 quantum_backend: str = "qasm_simulator",
                 quantum_shots: int = 1024,
                 conv_kernel: int | None = None,
                 conv_threshold: float = 0.5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_quantum = use_quantum
        self.quantum_reps = quantum_reps
        self.quantum_backend = quantum_backend
        self.quantum_shots = quantum_shots
        self.conv_kernel = conv_kernel
        self.conv_threshold = conv_threshold

        # Pre‑processing with a quantum convolution filter if requested
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

        # Classical encoder
        enc_layers = []
        in_dim = self.preprocess_dim
        for hidden in self.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, hidden))
            enc_layers.append(nn.ReLU())
            enc_layers.append(nn.BatchNorm1d(hidden))
            if self.dropout > 0.0:
                enc_layers.append(nn.Dropout(self.dropout))
            in_dim = hidden
        enc_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Classical decoder
        dec_layers = []
        in_dim = self.latent_dim
        for hidden in reversed(self.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, hidden))
            dec_layers.append(nn.ReLU())
            dec_layers.append(nn.BatchNorm1d(hidden))
            if self.dropout > 0.0:
                dec_layers.append(nn.Dropout(self.dropout))
            in_dim = hidden
        dec_layers.append(nn.Linear(in_dim, self.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        # Quantum encoder: a variational circuit that refines the latent vector
        if self.use_quantum:
            self._build_quantum_encoder()

    def _build_quantum_encoder(self):
        """Construct a SamplerQNN that refines a classical latent vector."""
        # Quantum circuit with one qubit per latent dimension
        qr = QuantumRegister(self.latent_dim, "q")
        qc = QuantumCircuit(qr)

        # Parameters that encode the classical latent vector
        self.input_params = [Parameter(f"x{i}") for i in range(self.latent_dim)]
        for i, theta in enumerate(self.input_params):
            qc.rx(theta, qr[i])

        # Variational ansatz
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
        classical_latent = self.encoder(processed)
        if not self.use_quantum:
            return classical_latent
        # Refine the latent vector with a quantum circuit
        qt_latent = self.quantum_encoder(classical_latent.numpy(),
                                         return_type="numpy")
        return torch.tensor(qt_latent, dtype=classical_latent.dtype,
                            device=classical_latent.device)

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
        """Simple reconstruction training loop.

        Parameters
        ----------
        data : torch.Tensor
            The training data of shape (N, input_dim).
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        dataset = TensorDataset(data.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


__all__ = ["AutoencoderGen"]
