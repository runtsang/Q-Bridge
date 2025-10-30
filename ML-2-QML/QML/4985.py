import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN as QEstimatorQNN
from qiskit_machine_learning.primitives import StatevectorEstimator
from qiskit.circuit.library import RealAmplitudes

class QuantumSelfAttention:
    """
    Builds a tiny self‑attention style circuit that can be embedded into the
    variational core.  Parameters are split into a rotation block (3 per qubit)
    and an entangling block (one CRX per qubit pair).
    """
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.qr = QuantumRegister(n_qubits, "q")
        self.cr = ClassicalRegister(n_qubits, "c")

    def _circuit(self, rot_params: Sequence[float], ent_params: Sequence[float]) -> QuantumCircuit:
        qc = QuantumCircuit(self.qr, self.cr)
        for i in range(self.n_qubits):
            qc.rx(rot_params[3*i], i)
            qc.ry(rot_params[3*i+1], i)
            qc.rz(rot_params[3*i+2], i)
        for i in range(self.n_qubits-1):
            qc.crx(ent_params[i], i, i+1)
        qc.measure(self.qr, self.cr)
        return qc

    def run(self, backend, rot_params, ent_params, shots=1024):
        qc = self._circuit(rot_params, ent_params)
        job = backend.run(qc, shots=shots)
        return job.result().get_counts(qc)

def EstimatorQNN() -> QEstimatorQNN:
    """
    Quantum hybrid regressor that mirrors the classical EstimatorQNN:
      * A RealAmplitudes ansatz (3 reps) serves as the variational core.
      * Parameters are divided into a rotation block and an entangling block,
        forming a quantum self‑attention style embedding.
      * The expectation value of a single‑qubit Y observable over all qubits
        is returned as the scalar output.
    """
    num_qubits = 4
    ansatz = RealAmplitudes(num_qubits, reps=3)

    # Create parameter lists
    rot_params = [Parameter(f"r{i}") for i in range(num_qubits*3)]
    ent_params = [Parameter(f"e{i}") for i in range(num_qubits-1)]

    # Build the circuit
    qc = QuantumCircuit(num_qubits)
    qc.compose(ansatz, inplace=True)

    # Inject attention layer
    attention = QuantumSelfAttention(num_qubits)
    qc.compose(attention._circuit(rot_params, ent_params), inplace=True)
    qc.measure_all()

    # Define observable: sum of Y on each qubit
    obs = SparsePauliOp.from_list([("Y"*num_qubits, 1)])

    # Estimator primitive
    estimator = StatevectorEstimator()
    return QEstimatorQNN(
        circuit=qc,
        observables=obs,
        input_params=rot_params,
        weight_params=ent_params,
        estimator=estimator,
    )

__all__ = ["EstimatorQNN"]
