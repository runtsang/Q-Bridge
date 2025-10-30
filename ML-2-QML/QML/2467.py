"""Quantum sampler autoencoder circuit builder."""
from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class HybridQuantumSamplerAutoencoder:
    """
    Builds a domain‑wall augmented RealAmplitudes circuit that can be turned into
    a SamplerQNN for use in the hybrid model.
    """
    def __init__(self, num_latent: int, num_trash: int = 2) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        # Total qubits: latent + 2*trash + 1 auxiliary
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Parameterized ansatz on latent+trash qubits
        qc.compose(
            RealAmplitudes(self.num_latent + self.num_trash, reps=5),
            range(0, self.num_latent + self.num_trash),
            inplace=True,
        )
        qc.barrier()

        # Domain‑wall style swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def get_sampler_qnn(self, input_params: list | None = None) -> SamplerQNN:
        """
        Wraps the circuit in a SamplerQNN.  `input_params` should match the
        parameter vector used for classical conditioning (empty by default).
        """
        if input_params is None:
            input_params = []
        sampler = StatevectorSampler()
        qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=input_params,
            weight_params=self.circuit.parameters,
            sampler=sampler,
        )
        return qnn

__all__ = ["HybridQuantumSamplerAutoencoder"]
