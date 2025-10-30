import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """Quantum layer that maps a classical latent vector to a new latent vector using a variational circuit."""
    def __init__(self, latent_dim: int, num_trash: int = 2):
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.qnn = self._build_qnn()

    def _build_qnn(self) -> SamplerQNN:
        total_qubits = self.latent_dim + 2 * self.num_trash + 1
        qc = QuantumCircuit(total_qubits)

        # Feature map: Ry rotations parameterized by input data
        input_params = [Parameter(f"phi{i}") for i in range(self.latent_dim)]
        for i, p in enumerate(input_params):
            qc.ry(p, i)

        # Ansatz
        ansatz = RealAmplitudes(self.latent_dim + self.num_trash, reps=3)
        qc.append(ansatz, range(0, self.latent_dim + self.num_trash))

        # Swap test to compare trash qubits
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)

        # Measure all qubits
        qc.measure_all()

        sampler = Sampler()

        def interpret(out: dict[str, int]) -> np.ndarray:
            """Compute expectation value of each qubit (probability of |1>)."""
            probs = np.zeros(self.latent_dim)
            total_counts = sum(out.values())
            for bitstring, count in out.items():
                bits = np.array([int(b) for b in bitstring[::-1]])  # reverse order
                probs += bits * count
            probs /= total_counts
            return probs

        qnn = SamplerQNN(
            circuit=qc,
            input_params=input_params,
            weight_params=ansatz.parameters,
            interpret=interpret,
            output_shape=(self.latent_dim,),
            sampler=sampler,
        )
        return qnn

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run the quantum circuit on a batch of latent vectors."""
        param_dict = {p: x[:, idx].tolist() for idx, p in enumerate(self.qnn.input_params)}
        out = self.qnn.run(param_dict)
        return np.array(out)

__all__ = ["HybridAutoencoder"]
