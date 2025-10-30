import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit import Aer, execute
from qiskit.circuit.random import random_circuit
from qiskit.primitives import Sampler as StatevectorSampler

class HybridEstimatorQNN:
    """Quantum hybrid estimator combining quanvolution, autoencoder ansatz and measurement."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 3,
                 trash_dim: int = 2,
                 shots: int = 1000,
                 threshold: float = 0.5,
                 backend=None) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler(self.backend)
        # Prepare a SamplerQNN for fast batched evaluation
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=[p for p in self.circuit.parameters if p.name.startswith("theta")],
            interpret=lambda x: x,
            output_shape=1,
            sampler=self.sampler
        )

    def _build_circuit(self) -> QuantumCircuit:
        # Quanvolution part: encode each input feature as a rotation
        qr = QuantumRegister(self.input_dim, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        for i, qubit in enumerate(qr):
            angle = Parameter(f"phi_{i}")
            qc.ry(angle, qubit)

        # Random entangling layer to mix features
        qc += random_circuit(self.input_dim, 2)

        # Autoencoder ansatz with swap test
        num_qubits = self.latent_dim + 2 * self.trash_dim + 1
        auto_qr = QuantumRegister(num_qubits, "auto")
        auto_cr = ClassicalRegister(1, "auto_c")
        auto_qc = QuantumCircuit(auto_qr, auto_cr)

        auto_qc.compose(RealAmplitudes(num_qubits, reps=5), auto_qr, inplace=True)
        auto_qc.barrier()
        aux = num_qubits - 1
        for i in range(self.trash_dim):
            auto_qc.cswap(aux, i, self.latent_dim + i)

        auto_qc.h(aux)
        auto_qc.measure(aux, auto_cr[0])

        # Attach autoencoder circuit to main circuit
        qc.compose(auto_qc, [q for q in auto_qr], inplace=True)
        qc.measure_all()

        return qc

    def run(self, data: np.ndarray) -> float:
        """Evaluate the hybrid estimator on classical data."""
        # Bind input parameters
        bindings = {f"phi_{i}": float(val) for i, val in enumerate(data.flatten()[:self.input_dim])}
        bound_qc = self.circuit.assign_parameters(bindings, inplace=False)
        result = execute(bound_qc, self.backend, shots=self.shots).result()
        counts = result.get_counts(bound_qc)
        # Compute average probability of measuring |1> on the last qubit (swap test output)
        total = 0
        for bitstring, cnt in counts.items():
            if bitstring[-1] == '1':
                total += cnt
        return total / (self.shots * 1)

    def evaluate(self,
                 observables: Iterable[Callable[[np.ndarray], float]],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[float]]:
        results: List[List[float]] = []
        for params in parameter_sets:
            val = self.run(np.array(params))
            results.append([val])
        return results

__all__ = ["HybridEstimatorQNN"]
