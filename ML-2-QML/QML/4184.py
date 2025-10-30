"""Quantum implementation of a hybrid binary classifier that leverages a
quantum auto‑encoder and a SamplerQNN head to produce probability
distributions.  The architecture follows the classical counterpart but
replaces the dense head with a parameterised quantum circuit."""
import numpy as np
import torch
import torch.nn as nn
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

def build_autoencoder_circuit(num_latent: int = 2,
                              num_trash: int = 1) -> QuantumCircuit:
    """
    Build a minimal quantum auto‑encoder circuit with a swap‑test auxiliary
    qubit.  The first qubit is used to encode the input feature via a
    Ry gate; the remaining qubits are parameterised by a RealAmplitudes
    ansatz.  The auxiliary qubit is measured to obtain a probability
    distribution that will be interpreted as the class probability.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Input encoding: a single Ry gate on the first qubit.
    qc.ry("theta0", 0)

    # Ansatz on latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=3)
    qc.compose(ansatz, range(num_latent + num_trash), inplace=True)

    # Swap‑test with auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)

    qc.measure(aux, cr[0])
    return qc

class HybridQCNet(nn.Module):
    """
    Quantum hybrid classifier that encodes features into a quantum
    auto‑encoder and samples the auxiliary qubit to obtain a
    probability distribution.  A lightweight classical head maps the
    sampler output to logits for the final probability pair.
    """
    def __init__(self,
                 num_latent: int = 2,
                 num_trash: int = 1) -> None:
        super().__init__()
        self.circuit = build_autoencoder_circuit(num_latent, num_trash)
        # Sampler primitive to obtain measurement counts
        self.sampler = Sampler()
        # SamplerQNN wrapper that interprets the auxiliary qubit
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
            sampler=self.sampler,
            interpret=lambda x: x,
            output_shape=2,
        )
        # Classical linear head to map sampler output to a single logit
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass expects a batch of scalar input angles for the first
        qubit (shape: [batch_size, 1]).  The angles are bound to the
        circuit's parameter, the sampler is executed, and the resulting
        probability distribution is fed into the classical classifier.
        """
        batch_size = x.shape[0]
        # Prepare parameter bindings for each sample
        param_binds = [{self.circuit.parameters[0]: val.item()}
                       for val in x.squeeze()]
        # Execute sampler and retrieve counts
        counts = self.sampler_qnn.run(param_binds)
        # Convert counts to probabilities of measuring |1⟩
        probs = np.array([counts[i][0] for i in range(batch_size)])
        probs_tensor = torch.tensor(probs, dtype=torch.float32,
                                    device=x.device).unsqueeze(-1)
        logits = self.classifier(probs_tensor)
        probs_final = torch.cat([logits.sigmoid(), 1 - logits.sigmoid()], dim=-1)
        return probs_final

__all__ = ["HybridQCNet"]
