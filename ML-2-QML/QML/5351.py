import qiskit
import numpy as np
import torch
from qiskit import transpile, assemble
from qiskit.circuit import Parameter

class QuantumCircuit:
    """Quantum circuit that returns the expectation value of a Y observable
    for a single qubit, parameterised by a single angle."""
    def __init__(self, n_qubits: int, backend, shots: int):
        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.theta = Parameter("theta")
        all_qubits = list(range(n_qubits))
        self._circuit.h(all_qubits)
        self._circuit.barrier()
        self._circuit.ry(self.theta, all_qubits)
        self._circuit.measure_all()

        self.backend = backend
        self.shots = shots

    def run(self, angles: np.ndarray) -> np.ndarray:
        compiled = transpile(self._circuit, self.backend)
        qobj = assemble(
            compiled,
            shots=self.shots,
            parameter_binds=[{self.theta: a} for a in angles]
        )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        def expectation(counts):
            # Convert counts to a one‑digit state string
            output = list(counts.keys())[0]
            #... but we do a simple weighted sum over all states
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            return np.sum(states * probs)

        return np.array([expectation(r) for r in result])

class QuantumHybridLayer:
    """A torch.autograd.Function that works with the quantum circuit."""
    def __init__(self, n_qubits, backend, shots, shift=0.0):
        self.circuit = QuantumCircuit(n_qubits, backend, shots)
        self.shift = shift

    def forward(self, x: torch.Tensor):
        angles = torch.squeeze(x).float()
        return torch.tensor(self.circuit.run(angles.cpu().numpy()), device=x.device)
