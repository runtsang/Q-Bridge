import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.optimizers import COBYLA

class SelfAttentionHybrid:
    """
    Quantum counterpart of the classical SelfAttentionHybrid.
    The circuit fuses self‑attention style gates, a lightweight auto‑encoder sub‑circuit,
    and a parameterised regression head that maps measurement outcomes to a scalar.
    """
    def __init__(self, n_qubits: int, latent_dim: int = 4):
        self.n_qubits = n_qubits
        self.latent_dim = latent_dim
        self.rotation_params = ParameterVector('theta', length=3*n_qubits)
        self.entangle_params  = ParameterVector('phi',   length=n_qubits-1)
        self.base_circuit = self._build_base()
        self.weights = np.random.randn(n_qubits) * 0.1
        self.bias = 0.0
        self.backend = Aer.get_backend('qasm_simulator')

    def _build_base(self) -> QuantumCircuit:
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)

        # Self‑attention style rotations
        for i in range(self.n_qubits):
            qc.rx(self.rotation_params[3*i], i)
            qc.ry(self.rotation_params[3*i+1], i)
            qc.rz(self.rotation_params[3*i+2], i)

        # Entangling layer mimicking CRX attention
        for i in range(self.n_qubits-1):
            qc.crx(self.entangle_params[i], i, i+1)

        # Auto‑encoder style lightweight layer
        for i in range(self.n_qubits):
            qc.rx(0.1, i)

        qc.measure(qr, cr)
        return qc

    def run(self,
            rotation_values: np.ndarray,
            entangle_values: np.ndarray,
            shots: int = 1024) -> float:
        """
        Execute the parameterised circuit and return a scalar regression prediction.
        """
        param_bindings = {p: v for p, v in zip(self.rotation_params, rotation_values)}
        param_bindings.update({p: v for p, v in zip(self.entangle_params, entangle_values)})
        bound_qc = self.base_circuit.bind_parameters(param_bindings)

        job = execute(bound_qc, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_qc)

        # Convert counts to expectation of Pauli‑Z for each qubit
        exp_z = np.zeros(self.n_qubits)
        for state, cnt in counts.items():
            bits = np.array(list(map(int, state[::-1])))  # little‑endian
            parity = (-1)**bits.sum()
            exp_z += parity * cnt
        exp_z /= shots

        # Linear regression head
        pred = float(np.dot(self.weights, exp_z) + self.bias)
        return pred

    def train_weights(self,
                      rotation_data: np.ndarray,
                      entangle_data: np.ndarray,
                      targets: np.ndarray,
                      lr: float = 0.01,
                      epochs: int = 50):
        """
        Simple gradient‑free optimiser for the linear regression head.
        """
        opt = COBYLA(maxiter=epochs)
        def objective(params):
            self.weights = params[:self.n_qubits]
            self.bias = params[self.n_qubits]
            preds = np.array([self.run(r, e) for r, e in zip(rotation_data, entangle_data)])
            return np.mean((preds - targets)**2)
        initial = np.concatenate([self.weights, [self.bias]])
        opt.optimize(num_vars=len(initial), objective_function=objective, initial_point=initial)
