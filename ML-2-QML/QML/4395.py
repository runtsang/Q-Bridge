import torch
from torch import nn
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """
    Quantum‑enhanced auto‑encoder that uses a RealAmplitudes ansatz to
    encode the input, a swap‑test to compare it with a latent register,
    and a classical decoder to reconstruct the data.  The quantum
    portion is fully differentiable via a state‑vector sampler.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: tuple[int,...] = (128, 64),
                 dropout: float = 0.1,
                 reps: int = 3) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.reps = reps

        # --- Quantum encoder (swap‑test circuit) --------------------------------
        qr = QuantumRegister(input_dim + latent_dim)
        cr = ClassicalRegister(1)
        self.qc = QuantumCircuit(qr, cr)

        # Encode the raw input
        self.qc.append(RealAmplitudes(input_dim, reps=reps), qr[:input_dim])
        # Encode the latent register (initially zeros)
        self.qc.append(RealAmplitudes(latent_dim, reps=reps), qr[input_dim:])
        # Swap‑test to compare the two registers
        aux = qr[-1]
        self.qc.h(aux)
        for i in range(latent_dim):
            self.qc.cswap(aux, qr[i], qr[input_dim + i])
        self.qc.h(aux)
        self.qc.measure(aux, cr[0])

        # Sampler for state‑vector evaluation
        self.sampler = Sampler()

        # SamplerQNN wrapper – the weight parameters are the circuit's
        # RealAmplitudes parameters (both input and latent).
        self.qnn = SamplerQNN(
            circuit=self.qc,
            input_params=[],
            weight_params=[p for p in self.qc.parameters],
            interpret=lambda x: x,
            output_shape=latent_dim,
            sampler=self.sampler
        )

        # Classical decoder (identical to the one in the original Autoencoder)
        decoder_layers = []
        in_dim = latent_dim
        for hidden in hidden_dims[::-1]:
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_dim)
            Input data to be reconstructed.
        Returns
        -------
        torch.Tensor of shape (batch, input_dim)
            Reconstructed data.
        """
        # Run the quantum encoder for each sample
        latent = []
        for vec in x:
            # Build a copy of the circuit with the current input vector
            qc = self.qc.copy()
            # Assign parameters: first input_dim params are the input vector,
            # the remaining are left as trainable (latent).
            qc.set_parameters(vec.tolist() + [0.0] * self.latent_dim)
            result = self.sampler.run(qc)
            state = result.result().get_statevector()
            latent.append(state.real)
        latent = torch.tensor(latent, dtype=torch.float32, device=x.device)
        # Decode the latent representation
        return self.decoder(latent)

__all__ = ["HybridAutoencoder"]
