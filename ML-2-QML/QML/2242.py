import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from typing import Tuple

# --------------------------------------------------------------------------- #
# Variational photonic‑style circuit (quantum‑centric) ----------------------- #
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clip a value to keep parameters within a valid range."""
    return max(-bound, min(bound, value))

def quantum_latent_circuit(
    num_qubits: int,
    latent_dim: int,
    reps: int = 2,
) -> QuantumCircuit:
    """
    Builds a variational circuit that encodes a latent vector and
    outputs `latent_dim` measurement results.  The circuit is a
    hybrid of RealAmplitudes ansatz and simple photonic‑like gates
    (implemented with qubit gates for simulation purposes).
    """
    qc = QuantumCircuit(num_qubits)
    # Encode the latent vector into the first `latent_dim` qubits
    # using a RealAmplitudes ansatz (parameter‑free for demo)
    qc.append(RealAmplitudes(num_qubits, reps=reps), range(num_qubits))
    # Simple photonic‑style operations translated to qubit gates
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(0.1 * i, i)
    # Measurement of the first `latent_dim` qubits
    return qc

def build_quantum_latent_qnn(
    num_qubits: int,
    latent_dim: int,
    reps: int = 2,
) -> SamplerQNN:
    """
    Returns a SamplerQNN that maps a latent vector (via circuit parameters)
    to a `latent_dim`‑dimensional output.  The QNN uses a variational
    circuit and a state‑vector sampler.
    """
    qc = quantum_latent_circuit(num_qubits, latent_dim, reps=reps)
    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],  # no external classical input for demo
        weight_params=qc.parameters,
        interpret=lambda x: x,  # identity
        output_shape=(latent_dim,),
        sampler=sampler,
    )
    return qnn

class QuantumLatentExtractor:
    """
    Wrapper around the SamplerQNN that provides a PyTorch‑compatible
    forward method.  It can be integrated into a hybrid training loop.
    """
    def __init__(self, num_qubits: int, latent_dim: int, reps: int = 2):
        self.qnn = build_quantum_latent_qnn(num_qubits, latent_dim, reps)
        # Convert the QNN parameters to torch tensors for gradient flow
        self.params = nn.Parameter(torch.tensor(self.qnn.weight_params, dtype=torch.float32))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        # Convert latent to numpy array (no classical input used)
        # The QNN expects a 1‑D array of parameters; we treat the latent as weights
        # for demonstration; in a real setting you would embed the latent into the circuit.
        # Here we simply feed the latent as the circuit parameters.
        param_vals = latent.detach().cpu().numpy()
        # Ensure shape matches
        if param_vals.ndim == 1:
            param_vals = param_vals.reshape(1, -1)
        # Run the sampler
        result = self.qnn.evaluate(param_vals)
        # Convert to torch tensor
        return torch.tensor(result, dtype=torch.float32)

__all__ = [
    "quantum_latent_circuit",
    "build_quantum_latent_qnn",
    "QuantumLatentExtractor",
]
