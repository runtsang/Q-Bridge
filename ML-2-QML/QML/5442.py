import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

class HybridSelfAttention:
    """
    Quantum hybrid self‑attention that mirrors the classical module.
    It encodes 2×2 patches into qubits, applies parameterised rotations
    and entangling gates, measures to obtain a distribution, and
    uses state‑fidelity‑based adjacency to weight the outputs before
    returning a scalar prediction.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 patch_size: int = 2,
                 adjacency_threshold: float = 0.8,
                 backend=None):
        self.n_qubits = n_qubits
        self.patch_size = patch_size
        self.adj_threshold = adjacency_threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.state_backend = Aer.get_backend("statevector_simulator")

        # Parameterised rotation angles
        self.theta = Parameter("θ")
        self.phi = Parameter("φ")

        self.circuit_template = self._build_circuit_template()

    def _build_circuit_template(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        # Encode patch via Ry rotations
        for i in range(self.n_qubits):
            qc.ry(self.theta, i)
        # Entangling block
        for i in range(self.n_qubits - 1):
            qc.crx(self.phi, i, i + 1)
        qc.measure(qr, cr)
        return qc

    def _encode_patch_circuit(self, patch: np.ndarray) -> QuantumCircuit:
        """
        Create a circuit instance for a single 2×2 patch.
        The patch values (0–1) are mapped to rotation angles in [0,π].
        """
        qc = self.circuit_template.copy()
        angles = patch.flatten() * np.pi
        params = {self.theta: angles[0], self.phi: angles[1]}
        qc.assign_parameters(params, inplace=True)
        return qc

    def _state_fidelity(self, psi: Statevector, phi: Statevector) -> float:
        return abs((psi.dag() @ phi)[0, 0]) ** 2

    def run(self, image: np.ndarray, shots: int = 1024) -> float:
        """
        Execute the self‑attention over all patches of a 28×28 image
        and return a scalar prediction.
        """
        # 1. split image into 2×2 patches
        patches = []
        for r in range(0, 28, self.patch_size):
            for c in range(0, 28, self.patch_size):
                patch = image[r:r + self.patch_size, c:c + self.patch_size]
                patches.append(patch)

        # 2. build circuits for each patch
        circuits = [self._encode_patch_circuit(p) for p in patches]

        # 3. simulate to obtain statevectors
        states = [Statevector.from_instruction(c, backend=self.state_backend) for c in circuits]

        # 4. compute pairwise fidelities and adjacency mask
        n = len(states)
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                fid = self._state_fidelity(states[i], states[j])
                adjacency[i, j] = 1.0 if fid >= self.adj_threshold else 0.0

        # 5. run measurement to get probability distribution per patch
        job = execute(circuits, self.backend, shots=shots)
        result = job.result()
        probs = np.zeros(n)
        for idx, circ in enumerate(circuits):
            counts = result.get_counts(circ)
            total = sum(counts.values())
            # probability of measuring the last qubit in state |1>
            prob = sum(v for k, v in counts.items() if k[-1] == '1') / total
            probs[idx] = prob

        # 6. weighted attention: adjacency * probs
        weighted = adjacency @ probs

        # 7. return scalar prediction as weighted sum
        return float(weighted.sum())
