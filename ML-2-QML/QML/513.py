import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Hybrid quantum‑classical self‑attention block.

    The circuit consists of a layer of single‑qubit rotations followed by a
    nearest‑neighbour CRX entanglement pattern.  Parameters are passed in the
    same order as the classical version so the two interfaces stay
    interchangeable.  The class exposes two execution modes:

    * ``run`` – returns a dictionary of measurement counts (original seed).
    * ``expectation`` – returns the expectation values of Pauli‑Z on each qubit
      after the rotation layer; these values can be interpreted as attention
      scores and are useful for gradient‑based optimisation.
    """
    def __init__(self, n_qubits: int = 4, shots: int = 1024, device=None):
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = device or qml.device("default.qubit", wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entanglement layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            # Return expectation of Pauli‑Z for each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        return circuit

    def run(self,
            backend,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = None,
            return_counts: bool = True):
        """
        Execute the circuit on the supplied backend.

        Parameters
        ----------
        backend : qiskit or pennylane backend
            In the original seed a Qiskit Aer simulator was used.  The method
            accepts either a Pennylane device or a Qiskit backend so the
            implementation is agnostic to the underlying provider.
        rotation_params, entangle_params : np.ndarray
            Parameters passed identically to the classical version.
        shots : int, optional
            Number of shots for a probabilistic execution.  If ``None`` the
            default stored in ``self.shots`` is used.
        return_counts : bool
            If ``True`` the method returns a dictionary of measurement counts
            (mimicking the original Qiskit behaviour).  If ``False`` the
            expectation values are returned.

        Returns
        -------
        dict | np.ndarray
            Either the counts dictionary or a 1‑D array of expectation values.
        """
        shots = shots or self.shots
        # Pennylane path – deterministic expectation values
        if isinstance(backend, qml.Device):
            circuit = self._circuit(rotation_params, entangle_params)
            return np.array(circuit())
        # Qiskit path – simulate with the provided backend
        else:
            import qiskit
            from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute

            qr = QuantumRegister(self.n_qubits, "q")
            cr = ClassicalRegister(self.n_qubits, "c")
            qc = QuantumCircuit(qr, cr)

            for i in range(self.n_qubits):
                qc.rx(rotation_params[3 * i], i)
                qc.ry(rotation_params[3 * i + 1], i)
                qc.rz(rotation_params[3 * i + 2], i)
            for i in range(self.n_qubits - 1):
                qc.crx(entangle_params[i], i, i + 1)
            qc.measure(qr, cr)

            job = execute(qc, backend, shots=shots)
            result = job.result()
            if return_counts:
                return result.get_counts(qc)
            else:
                # Convert counts to expectation values
                counts = result.get_counts(qc)
                probs = np.array([counts.get(bit, 0) for bit in result.get_counts(qc).keys()]) / shots
                exps = []
                for i in range(self.n_qubits):
                    mask = 1 << (self.n_qubits - 1 - i)
                    exps.append(np.sum(probs * np.where(np.array([int(b, 2) & mask!= 0 for b in result.get_counts(qc).keys()]), -1, 1)))
                return np.array(exps)
