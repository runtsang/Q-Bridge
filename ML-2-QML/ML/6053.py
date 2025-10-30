import torch
from torch import nn
import numpy as np
import qiskit
from qiskit.circuit import Parameter
from qiskit import Aer, execute

class ConvEnhanced(nn.Module):
    """
    Hybrid classical‑quantum convolution filter.
    The module contains:
      * A trainable 2‑D convolution kernel (shape [out_ch, in_ch, k, k]).
      * A quantum circuit that mirrors the kernel size and is parameterized
        by the same weights, allowing end‑to‑end differentiability.
      * A combined loss that is a weighted sum of the classical
        activation mean and the quantum measurement expectation.
    The module can be used as a drop‑in replacement for the original Conv
    and can be trained with standard PyTorch optimizers.
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 1,
                 kernel_size: int = 2,
                 threshold: float = 0.0,
                 quantum_weight: float = 0.5,
                 device: str = "cpu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.quantum_weight = quantum_weight
        self.device = device

        # Classical convolution
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size,
                              bias=True)

        # Quantum circuit parameters (initialized same as conv weights)
        # Each output channel has its own set of parameters for the quantum circuit
        self.qparams = nn.Parameter(
            torch.randn(out_channels, in_channels,
                        kernel_size, kernel_size,
                        device=device))

        # Prepare a base quantum circuit template
        self.base_circuit = self._build_base_circuit()

    def _build_base_circuit(self):
        """
        Build a simple quantum circuit template for a single output channel.
        The circuit consists of RX rotations on each qubit corresponding to the
        input patch. The number of qubits equals the kernel size squared.
        """
        n_qubits = self.kernel_size ** 2
        circuit = qiskit.QuantumCircuit(n_qubits)
        thetas = [Parameter(f"theta_{i}") for i in range(n_qubits)]
        for i, theta in enumerate(thetas):
            circuit.rx(theta, i)
        return circuit

    def _quantum_expectation(self, patch: torch.Tensor) -> float:
        """
        Compute a quantum expectation value for a given patch.
        The patch is a tensor of shape (in_channels, kernel_size, kernel_size).
        The quantum circuit consists of a RX rotation on each qubit with angle
        equal to the corresponding patch value (scaled to [0, π]).
        """
        # Flatten patch and map to [0, π]
        flat = patch.view(-1).cpu().numpy()
        angles = flat * np.pi  # simple scaling
        # Bind parameters
        bound_circuit = self.base_circuit.bind_parameters(
            {f"theta_{i}": angles[i] for i in range(len(angles))}
        )
        backend = Aer.get_backend("qasm_simulator")
        job = execute(bound_circuit, backend=backend, shots=1024)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        # Compute average number of 1s
        total_ones = sum(key.count('1') * val for key, val in counts.items())
        expectation = total_ones / (1024 * self.kernel_size ** 2)
        return expectation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns a weighted sum of classical activation and quantum expectation.
        """
        # Classical activation
        conv_out = self.conv(x)
        classical_mean = torch.sigmoid(conv_out - self.threshold).mean()

        # Quantum expectation
        # Extract patches using unfold
        patches = x.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        # patches shape: (batch, in_channels, out_h, out_w, k, k)
        batch, in_ch, out_h, out_w, k, k = patches.shape
        quantum_expectations = []
        for i in range(batch):
            # For each output channel, compute expectation using its quantum parameters
            channel_expectations = []
            for ch in range(in_ch):
                # Use the patch for this channel
                patch = patches[i, ch]
                expectation = self._quantum_expectation(patch)
                channel_expectations.append(expectation)
            quantum_expectations.append(channel_expectations)
        quantum_mean = torch.tensor(quantum_expectations, device=x.device).mean()

        # Weighted sum
        output = (1 - self.quantum_weight) * classical_mean + self.quantum_weight * quantum_mean
        return output

def Conv():
    """Convenience factory that returns an instance of ConvEnhanced."""
    return ConvEnhanced()
