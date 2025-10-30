"""Quantum autoencoder with a self‑attention style ansatz.

The circuit first prepares a feature sub‑state with a
RealAmplitudes ansatz.  A second block implements a
self‑attention‑like coupling between latent qubits and
auxiliary trash qubits using controlled‑X gates.  The
entire construction is wrapped in a SamplerQNN so that
classical optimization can be performed on the circuit
parameters.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN


def _build_attention_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a circuit that encodes a self‑attention style block."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Feature encoding with a shallow RealAmplitudes ansatz
    circuit.append(RealAmplitudes(num_latent + num_trash, reps=3), range(num_latent + num_trash))

    # Controlled‑X coupling between latent and trash qubits
    for i in range(num_trash):
        circuit.cx(num_latent + i, num_latent + num_trash + i)

    # Swap‑test style measurement of the auxiliary qubit
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit


class AutoencoderGen155:
    """Quantum autoencoder that exposes a SamplerQNN interface."""
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        sampler: Sampler | None = None,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.sampler = sampler or Sampler()
        self.circuit = _build_attention_circuit(num_latent, num_trash)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def predict(self, inputs: np.ndarray, shots: int = 1024) -> np.ndarray:
        """Run the sampler and return the probability distribution."""
        # In a full model the inputs would be encoded into the circuit
        # via parameterization.  Here we simply forward the raw
        # parameters to the SamplerQNN for demonstration.
        return self.qnn.predict(inputs, shots=shots)


def create_AutoencoderGen155(
    num_latent: int = 3,
    num_trash: int = 2,
    sampler: Sampler | None = None,
) -> AutoencoderGen155:
    """Factory that returns a ready‑to‑use quantum autoencoder."""
    return AutoencoderGen155(num_latent=num_latent, num_trash=num_trash, sampler=sampler)


__all__ = ["AutoencoderGen155", "create_AutoencoderGen155"]
