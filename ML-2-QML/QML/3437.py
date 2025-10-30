import numpy as np
import pennylane as qml
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer

class HybridAttentionModel:
    """
    Quantum self‑attention block implemented with a variational circuit.
    The circuit uses the same parameter layout as the classical module:
    rotation_params shape (embed_dim, 3) for RX,RZ,RY per qubit and
    entangle_params shape (embed_dim-1,) for CRX gates between neighbours.
    The circuit can be executed on a Qiskit backend or the Pennylane
    simulator; the run() method returns measurement counts for the
    first qubit which act as attention logits.
    """

    def __init__(self, embed_dim: int, backend: str = "qasm_simulator"):
        self.embed_dim = embed_dim
        self.backend = backend
        self.dev = qml.device("default.qubit", wires=embed_dim)
        # Random initial parameters – will be overwritten by run()
        self.rotation_params = np.random.randn(embed_dim, 3)
        self.entangle_params = np.random.randn(embed_dim - 1)

    def _build_pennylane_circuit(self, rotation_params, entangle_params, inputs):
        @qml.qnode(self.dev)
        def circuit(inputs, rotation_params, entangle_params):
            # Encode inputs via angle encoding on each qubit
            for i, x in enumerate(inputs):
                qml.RX(np.pi * x, wires=i)
            # Rotation block
            for i in range(self.embed_dim):
                qml.RX(rotation_params[i, 0], wires=i)
                qml.RY(rotation_params[i, 1], wires=i)
                qml.RZ(rotation_params[i, 2], wires=i)
            # Entanglement block – use CRX as a variational analog of CRX
            for i in range(self.embed_dim - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Measure expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))
        return circuit(inputs, rotation_params, entangle_params)

    def _build_qiskit_circuit(self, rotation_params, entangle_params, inputs):
        qr = QuantumRegister(self.embed_dim, "q")
        cr = ClassicalRegister(1, "c")
        circ = QuantumCircuit(qr, cr)

        # Encode inputs (simple angle encoding)
        for i, x in enumerate(inputs):
            circ.rx(np.pi * x, qr[i])

        # Rotation gates
        for i in range(self.embed_dim):
            circ.rx(rotation_params[i, 0], qr[i])
            circ.ry(rotation_params[i, 1], qr[i])
            circ.rz(rotation_params[i, 2], qr[i])

        # Entanglement gates
        for i in range(self.embed_dim - 1):
            circ.crx(entangle_params[i], qr[i], qr[i + 1])

        # Measurement on qubit 0
        circ.measure(qr[0], cr[0])
        return circ

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """
        Execute the variational attention circuit.

        Args:
            rotation_params: (embed_dim, 3) array of RX/RY/RZ angles.
            entangle_params: (embed_dim-1,) array of CRX angles.
            inputs: (embed_dim,) array of feature values in [0,1].
            shots: number of measurement shots.

        Returns:
            Dictionary of measurement counts for qubit 0, i.e.
            {'0': count_0, '1': count_1} representing the attention logits.
        """
        # Clip parameters to keep the circuit stable
        rotation_params = np.clip(rotation_params, -5.0, 5.0)
        entangle_params = np.clip(entangle_params, -5.0, 5.0)

        circ = self._build_qiskit_circuit(rotation_params, entangle_params, inputs)

        backend = Aer.get_backend(self.backend)
        job = execute(circ, backend=backend, shots=shots)
        return job.result().get_counts(circ)

    def run_pennylane(self, rotation_params, entangle_params, inputs):
        """
        Optional Pennylane execution path – returns the expectation
        value of PauliZ on qubit 0 as a continuous attention score.
        """
        return self._build_pennylane_circuit(rotation_params, entangle_params, inputs)
