import numpy as np
import pennylane as qml
import pennylane.numpy as pnp

class SelfAttention:
    """
    Quantum‑classical Self‑Attention block.
    """
    def __init__(self, embed_dim: int, n_qubits: int | None = None):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits or embed_dim
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev)
        def circuit(r, e):
            for i in range(self.n_qubits):
                qml.RX(r[3 * i], wires=i)
                qml.RY(r[3 * i + 1], wires=i)
                qml.RZ(r[3 * i + 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(e[i], wires=[i, i + 1])
            return qml.expval(qml.PauliZ(wires=list(range(self.n_qubits))))

        self.circuit = circuit

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            backend=None,
            shots: int = 1024) -> np.ndarray:
        """
        Compute hybrid attention.
        If backend is None, uses Pennylane simulator.
        If backend is a Qiskit backend, executes the equivalent circuit.
        """
        B, L, D = inputs.shape
        Q = inputs @ np.eye(D)
        K = inputs @ np.eye(D)
        V = inputs @ np.eye(D)
        scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(D)

        if backend is None:
            weights = self.circuit(rotation_params, entangle_params)
        else:
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
            qr = QuantumRegister(self.n_qubits, "q")
            cr = ClassicalRegister(self.n_qubits, "c")
            qc = QuantumCircuit(qr, cr)
            for i in range(self.n_qubits):
                qc.rx(rotation_params[3 * i], qr[i])
                qc.ry(rotation_params[3 * i + 1], qr[i])
                qc.rz(rotation_params[3 * i + 2], qr[i])
            for i in range(self.n_qubits - 1):
                qc.crx(entangle_params[i], qr[i], qr[i + 1])
            qc.measure(qr, cr)
            job = backend.execute(qc, shots=shots)
            counts = job.result().get_counts(qc)
            probs = np.array([counts.get(f"{i:0{self.n_qubits}b}", 0) for i in range(2**self.n_qubits)])
            probs = probs / probs.sum()
            weights = probs[:self.n_qubits]

        weights = weights.reshape(1, 1, -1)
        scores = scores * weights
        attn = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        out = np.matmul(attn, V)
        return out
