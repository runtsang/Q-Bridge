import torch
from torch import nn
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply a domain‑wall pattern to qubits in the range [start, end)."""
    for i in range(start, end):
        circuit.x(i)
    return circuit

class QuantumLatentLayer(nn.Module):
    """
    Variational quantum circuit that maps a classical latent vector into a quantum state
    and returns a new latent representation via a SamplerQNN.  The circuit consists of:
    * a RealAmplitudes ansatz on the first `latent_dim` qubits,
    * a swap‑test style interaction with `trash_dim` auxiliary qubits,
    * optional measurement of an auxiliary qubit (used only for visualisation).
    """
    def __init__(self, latent_dim: int, trash_dim: int = 2, reps: int = 1, shots: int = 1024):
        super().__init__()
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.shots = shots

        # Build the circuit
        self.circuit = self._build_circuit()

        # SamplerQNN that interprets the first `latent_dim` amplitudes as outputs
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=list(self.circuit.parameters),
            interpret=lambda x: np.real(x[:latent_dim]),
            output_shape=latent_dim,
            sampler=Sampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        qr = QuantumRegister(num_qubits)
        qc = QuantumCircuit(qr)

        # Variational ansatz on the first `latent_dim` qubits
        ansatz = RealAmplitudes(self.latent_dim, reps=self.reps)
        qc.compose(ansatz, range(self.latent_dim), inplace=True)

        # Swap‑test style interaction with trash qubits
        qc.barrier()
        aux = self.latent_dim + 2 * self.trash_dim  # auxiliary qubit for swap test
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, i, self.latent_dim + i)
        qc.h(aux)

        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical latent vector of shape (..., latent_dim)

        Returns
        -------
        torch.Tensor
            Quantum‑processed latent vector of the same shape.
        """
        params = x.detach().cpu().numpy()
        outputs = self.qnn.forward(params)
        return torch.from_numpy(outputs).to(x.device).float()

def create_quantum_latent_layer(latent_dim: int, trash_dim: int = 2, reps: int = 1, shots: int = 1024) -> QuantumLatentLayer:
    """Convenience factory for a quantum latent layer."""
    return QuantumLatentLayer(latent_dim, trash_dim, reps, shots)

__all__ = ["QuantumLatentLayer", "create_quantum_latent_layer", "domain_wall"]
