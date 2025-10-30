"""
Quantum counterpart that implements the same logical flow using Qiskit primitives.
Each classical block is replaced by a parameterised circuit or a sampler‑QNN.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, RandomCircuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators import PauliZ

# --------------------------------------------------------------------------- #
# Helper: feature‑map circuit (CNN‑inspired)
# --------------------------------------------------------------------------- #
def feature_map_circuit(num_qubits: int) -> QuantumCircuit:
    """
    Encodes a classical image slice into a quantum state via
    a RealAmplitudes layer that mimics the convolutional feature extractor.
    """
    circ = QuantumCircuit(num_qubits)
    circ.compose(RealAmplitudes(num_qubits, reps=2), inplace=True)
    return circ

# --------------------------------------------------------------------------- #
# Quantum‑inspired variational block
# --------------------------------------------------------------------------- #
def variational_block(num_qubits: int) -> QuantumCircuit:
    """
    Random layer followed by a few RX/RZ gates – a minimal surrogate for the
    torchquantum QLayer in the seed.
    """
    circ = QuantumCircuit(num_qubits)
    circ.compose(RandomCircuit(num_qubits, depth=3), inplace=True)
    circ.rx(np.pi/4, 0)
    circ.rz(np.pi/3, 1)
    circ.cx(0, 1)
    circ.rx(np.pi/6, 2)
    return circ

# --------------------------------------------------------------------------- #
# Auto‑encoder style circuit (swap‑test based)
# --------------------------------------------------------------------------- #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Implements a simple swap‑test auto‑encoder that measures a single qubit
    to produce a scalar fidelity estimate (used as a latent score).
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circ = QuantumCircuit(qr, cr)

    # Encode latent + trash
    circ.compose(RealAmplitudes(num_latent + num_trash, reps=3), range(0, num_latent + num_trash), inplace=True)
    circ.barrier()

    # Swap test
    aux = num_latent + 2 * num_trash
    circ.h(aux)
    for i in range(num_trash):
        circ.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circ.h(aux)
    circ.measure(aux, cr[0])
    return circ

# --------------------------------------------------------------------------- #
# Sampler‑QNN wrapper
# --------------------------------------------------------------------------- #
class QuantumNATHybridQML:
    """
    Quantum implementation of the hybrid architecture.  Input is a 28×28 image
    flattened to a 784‑dim vector that is split into groups of 4 qubits
    (i.e. 196 qubits total).  The circuit is a concatenation of:
        1. Feature‑map block per group
        2. Variational block
        3. Auto‑encoder block
    The final state is sampled by a SamplerQNN to produce a probability distribution
    over the 4 class labels.
    """

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.sampler = Sampler()
        # Build a single group circuit and tile it 4 times
        group_circ = feature_map_circuit(4)
        group_circ.compose(variational_block(4), inplace=True)
        group_circ.compose(autoencoder_circuit(2, 1), inplace=True)
        # Repeat the group circuit
        self.circuit = QuantumCircuit(4 * 4)
        for i in range(4):
            self.circuit.compose(group_circ, range(4 * i, 4 * (i + 1)), inplace=True)

        # SamplerQNN mapping 4‑bit output to class logits
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=self.num_classes,
            sampler=self.sampler
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass a batch of flattened images through the quantum circuit.
        Returns a tensor of shape (batch, num_classes) with class probabilities.
        """
        # Convert to numpy and pad to required qubit count
        batch = x.detach().cpu().numpy()
        # Randomly initialise parameters for demonstration (real training would optimise them)
        param_values = np.random.rand(len(self.circuit.parameters))
        probs = self.qnn.predict(batch, param_values)
        return torch.from_numpy(np.array(probs))

__all__ = ["QuantumNATHybridQML"]
