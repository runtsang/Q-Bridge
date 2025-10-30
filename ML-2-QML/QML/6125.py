import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute

class SelfAttentionEnhanced:
    """
    Quantum self‑attention block implemented with a variational circuit.
    Supports parameter‑shift gradient evaluation and an optional
    measurement‑based attention map.
    """
    def __init__(self, n_qubits: int, use_attention_map: bool = False):
        self.n_qubits = n_qubits
        self.use_attention_map = use_attention_map
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")
        self.backend = Aer.get_backend("qasm_simulator")

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        circuit = QuantumCircuit(self.qr, self.cr)

        # Rotation layer: RX, RY, RZ per qubit
        for i in range(self.n_qubits):
            circuit.rx(rotation_params[3 * i], i)
            circuit.ry(rotation_params[3 * i + 1], i)
            circuit.rz(rotation_params[3 * i + 2], i)

        # Entanglement layer: controlled‑RX
        for i in range(self.n_qubits - 1):
            circuit.crx(entangle_params[i], i, i + 1)

        # Measurement handling
        if self.use_attention_map:
            circuit.measure(self.qr, self.cr)
        else:
            pass  # measurement added in run if needed

        return circuit

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            shots: int = 1024):
        """
        Execute the self‑attention circuit and return a probability
        distribution over basis states. If `use_attention_map` is True,
        the returned dict is interpreted as an attention map.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        if not self.use_attention_map:
            circuit.measure(self.qr, self.cr)

        job = execute(circuit, self.backend, shots=shots)
        result = job.result()
        return result.get_counts(circuit)

    def parameter_shift_gradient(self, rotation_params: np.ndarray,
                                 entangle_params: np.ndarray,
                                 loss_fn, shots: int = 1024,
                                 step: float = np.pi / 2):
        """
        Compute gradients of a loss function with respect to all parameters
        using the parameter‑shift rule.

        Parameters
        ----------
        rotation_params : np.ndarray
            Current rotation parameters.
        entangle_params : np.ndarray
            Current entanglement parameters.
        loss_fn : Callable[[dict], float]
            Function that maps measurement counts to a scalar loss.
        shots : int
            Number of shots for each circuit evaluation.
        step : float
            Shift value for the parameter‑shift rule (default π/2).
        """
        grads = np.zeros_like(rotation_params)
        # Gradients w.r.t rotation parameters
        for idx in range(len(rotation_params)):
            shift_plus = rotation_params.copy()
            shift_minus = rotation_params.copy()
            shift_plus[idx]  += step
            shift_minus[idx] -= step

            counts_plus  = self.run(shift_plus, entangle_params, shots)
            counts_minus = self.run(shift_minus, entangle_params, shots)

            loss_plus  = loss_fn(counts_plus)
            loss_minus = loss_fn(counts_minus)

            grads[idx] = 0.5 * (loss_plus - loss_minus)

        # Gradients w.r.t entanglement parameters
        for idx in range(len(entangle_params)):
            shift_plus = entangle_params.copy()
            shift_minus = entangle_params.copy()
            shift_plus[idx]  += step
            shift_minus[idx] -= step

            counts_plus  = self.run(rotation_params, shift_plus, shots)
            counts_minus = self.run(rotation_params, shift_minus, shots)

            loss_plus  = loss_fn(counts_plus)
            loss_minus = loss_fn(counts_minus)

            grads[idx + len(rotation_params)] = 0.5 * (loss_plus - loss_minus)

        return grads
