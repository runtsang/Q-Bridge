import torch
import torch.nn as nn
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch convolution producing 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

def auto_encoder_circuit(num_latent: int, num_trash: int) -> tuple[QuantumCircuit, list[Parameter]]:
    """Builds a quantum auto‑encoder circuit with a swap‑test and a RealAmplitudes ansatz."""
    num_qubits = num_latent + 2 * num_trash + 1
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_latent + 1, "c")
    qc = QuantumCircuit(qr, cr)

    # Parameterised Ry gates to encode the classical input
    input_params = [Parameter(f"θ_{i}") for i in range(num_trash)]
    for i, p in enumerate(input_params):
        qc.ry(p, qr[i])

    # Ansatz for the latent subspace
    ansatz = RealAmplitudes(num_latent, reps=5)
    qc.compose(ansatz, qr[num_trash : num_trash + num_latent], inplace=True)

    # Swap‑test between the trash and latent registers
    aux = qr[-1]
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, qr[num_trash + i], qr[num_trash + num_trash + i])
    qc.h(aux)

    # Measure the auxiliary qubit (fidelity proxy)
    qc.measure(aux, cr[0])

    # Measure each latent qubit to obtain a classical latent vector
    for i in range(num_latent):
        qc.measure(qr[num_trash + i], cr[i + 1])

    return qc, input_params

class QuantumAutoencoder(nn.Module):
    """Quantum auto‑encoder head based on a SamplerQNN."""
    def __init__(self, num_latent: int = 32, num_trash: int = 8) -> None:
        super().__init__()
        circuit, input_params = auto_encoder_circuit(num_latent, num_trash)
        weight_params = [p for p in circuit.parameters if p not in input_params]
        self.qnn = SamplerQNN(
            circuit=circuit,
            input_params=input_params,
            weight_params=weight_params,
            interpret=lambda x: x,
            output_shape=(num_latent,),
            sampler=Sampler()
        )
        self.num_trash = num_trash
        self.project = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.project is None:
            self.project = nn.Linear(x.size(1), self.num_trash, bias=False).to(x.device)
        projected = self.project(x)
        return self.qnn(projected)

class QuanvolutionAutoencoder(nn.Module):
    """Hybrid network: classical quanvolution → quantum auto‑encoder → classifier."""
    def __init__(self, latent_dim: int = 32, num_trash: int = 8) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.autoencoder = QuantumAutoencoder(num_latent=latent_dim, num_trash=num_trash)
        self.classifier = nn.Linear(latent_dim, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        latent = self.autoencoder(features)
        logits = self.classifier(latent)
        return logits

__all__ = ["QuanvolutionFilter", "QuantumAutoencoder", "QuanvolutionAutoencoder"]
