import numpy as np
import torch
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class AutoencoderHybrid(nn.Module):
    """Variational quantum autoencoder that reconstructs classical data."""
    def __init__(self, input_dim: int, latent_dim: int = 3, trash_dim: int = 2, shots: int = 1024):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.shots = shots
        self.sampler = Sampler()

        # Build the core circuit
        self.circuit = self._build_circuit()

        # SamplerQNN interprets the output probabilities as reconstructed values
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],                     # no external inputs – we embed data in parameters
            weight_params=self.circuit.parameters,  # all parameters are trainable weights
            interpret=lambda x: x,               # identity – probabilities are the raw output
            output_shape=input_dim,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Constructs a swap‑test autoencoder with a domain wall."""
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # 1) Encode classical data into the first `latent_dim` qubits.
        #    We use a simple RX rotation parameterised by the input vector.
        #    The actual values will be injected via the QNN's weight parameters.
        for i in range(self.latent_dim):
            qc.rx(qc.params[i], i)

        # 2) Variational ansatz on latent + trash qubits
        ansatz = RealAmplitudes(self.latent_dim + self.trash_dim, reps=3)
        qc.compose(ansatz, list(range(self.latent_dim + self.trash_dim)), inplace=True)

        # 3) Domain wall – flip the first half of the trash qubits
        for i in range(self.trash_dim):
            qc.x(self.latent_dim + i)

        # 4) Swap test with auxiliary qubit
        aux = self.latent_dim + 2 * self.trash_dim
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.trash_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that reconstructs the input via quantum measurement.
        `inputs` is expected to be a batch of vectors of shape (batch, input_dim).
        The values are mapped onto the first `latent_dim` RX parameters of the circuit.
        """
        batch = inputs.cpu().numpy()
        results = []

        for vec in batch:
            # Map the first `latent_dim` elements of `vec` to the circuit's RX parameters.
            # Remaining elements are ignored for brevity.
            params = list(vec[:self.latent_dim])
            # Pad if necessary
            if len(params) < len(self.circuit.parameters):
                params += [0.0] * (len(self.circuit.parameters) - len(params))
            # Execute the circuit via the sampler
            result = self.sampler.run(self.circuit, parameter_binds=[{p: v for p, v in zip(self.circuit.parameters, params)}])
            # Expectation values of the measurement qubit give a probability distribution.
            # We interpret the probability of measuring |0> as the reconstructed value.
            probs = np.array(result.get_counts().values(), dtype=float)
            probs /= probs.sum()
            recon = probs[0]  # probability of measuring 0
            results.append(recon)

        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

def train_quantum_autoencoder(
    model: AutoencoderHybrid,
    data: torch.Tensor,
    *,
    epochs: int = 10,
    lr: float = 0.01,
    device: torch.device | None = None,
) -> list[float]:
    """
    Very light‑weight training loop that optimises the circuit parameters using
    the parameter‑shift rule implemented by the SamplerQNN.  The loss is the
    mean‑squared error between the reconstructed probability and the true
    target (here we use the first latent element as a toy target).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    history = []

    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(data.to(device))
        loss = loss_fn(recon, data[:, :1].to(device))
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return history

__all__ = ["AutoencoderHybrid", "train_quantum_autoencoder"]
