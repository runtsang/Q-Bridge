import numpy as np
import torch
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler

class HybridQuantumSampler:
    """
    Quantum sampler implementing a variational auto‑encoder style circuit.
    The circuit consists of a RealAmplitudes ansatz, a domain‑wall preparation,
    and a swap‑test measurement yielding a two‑dimensional probability vector.
    The latent vector from the classical encoder is mapped to the ansatz weights.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2, repetitions: int = 5):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.repetitions = repetitions

        # Build ansatz once; parameters will be bound later
        self.ansatz = RealAmplitudes(num_latent + num_trash, reps=repetitions)

        # Domain‑wall preparation
        self.domain_wall = self._build_domain_wall(num_latent + num_trash)

        # Full circuit with swap‑test
        self.circuit = self._build_full_circuit()

        # Sampler primitive
        self.sampler = StatevectorSampler()

    def _build_domain_wall(self, size: int) -> QuantumCircuit:
        qc = QuantumCircuit(size)
        for i in range(size // 2, size):
            qc.x(i)
        return qc

    def _build_full_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Apply ansatz on latent + trash qubits
        qc.append(self.ansatz, range(0, self.num_latent + self.num_trash))

        # Swap‑test with auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def sample(self, latent: torch.Tensor | np.ndarray) -> np.ndarray:
        """
        Interpret the latent vector as weight parameters for the ansatz,
        bind them, run the sampler and return a 2‑element probability vector.
        """
        # Convert to numpy and ensure shape
        if isinstance(latent, torch.Tensor):
            latent = latent.detach().cpu().numpy()
        weights = latent.reshape(-1)

        # Bind parameters
        bound_qc = self.circuit.bind_parameters(
            {p: w for p, w in zip(self.ansatz.parameters, weights)}
        )

        # Execute sampler
        result = self.sampler.run(bound_qc, shots=1024).result()
        counts = result.get_counts()
        probs = np.zeros(2)
        probs[0] = counts.get("0", 0) / 1024
        probs[1] = counts.get("1", 0) / 1024
        return probs
