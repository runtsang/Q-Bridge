import numpy as np
import qiskit
from qiskit import assemble, transpile
from qiskit.circuit import ParameterVector

class HybridFullyConnectedLayer:
    """
    Quantum implementation of the hybrid fully‑connected layer.

    Features:
      * Parameterized Ry rotations (like the original FCL).
      * Two‑qubit entanglement (CX) to emulate SamplerQNN behaviour.
      * Photonic‑inspired scaling: expectation is scaled and shifted.
    """
    def __init__(self,
                 n_qubits: int = 2,
                 backend=None,
                 shots: int = 1024,
                 scale: float = 1.0,
                 shift: float = 0.0) -> None:
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.n_qubits = n_qubits
        self.scale = scale
        self.shift = shift

        self.theta = ParameterVector("theta", length=n_qubits)
        self.circuit = qiskit.QuantumCircuit(n_qubits)

        # Base layer: H → ry(θ) → CX (if n_qubits >= 2)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        for i, p in enumerate(self.theta):
            self.circuit.ry(p, i)
        if n_qubits >= 2:
            self.circuit.cx(0, 1)
        self.circuit.measure_all()

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the parameterised circuit and return the expectation of
        the computational basis measurement weighted by the binary state.
        """
        if len(thetas)!= self.n_qubits:
            raise ValueError("Theta length must match number of qubits.")
        bound = transpile(self.circuit, self.backend)
        qobj = assemble(
            bound,
            shots=self.shots,
            parameter_binds=[{p: t for p, t in zip(self.theta, thetas)}]
        )
        job = self.backend.run(qobj)
        result = job.result()
        counts = result.get_counts(self.circuit)
        expectation = self._expectation(counts)
        # Photonic‑style scaling
        return np.array([self.scale * expectation + self.shift])

    def _expectation(self, counts: dict) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        return np.sum(states * probs)

    def sampler_output(self, thetas: np.ndarray) -> np.ndarray:
        """
        Return a two‑element probability vector mimicking a softmax output,
        based on the quantum expectation.
        """
        prob = self.run(thetas)[0]
        return np.array([prob, 1 - prob])
