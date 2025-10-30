import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import Dgate, MeasureFock, Fock

class ConvFraudHybridQML:
    """Hybrid quantum model combining a variational convolution circuit and a photonic fraud detection circuit."""
    def __init__(self, kernel_size: int = 2, conv_shots: int = 500, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.conv_shots = conv_shots
        self.threshold = threshold
        self._build_conv_circuit()
        self.engine = sf.Engine("gaussian")

    def _build_conv_circuit(self):
        self.circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self.circuit.rx(self.theta[i], i)
        self.circuit.barrier()
        self.circuit += random_circuit(self.n_qubits, 2)
        self.circuit.measure_all()

    def _build_fraud_program(self, d1, d2):
        prog = sf.Program(2)
        with prog.context as q:
            Dgate(d1, 0) | q[0]
            Dgate(d2, 0) | q[1]
            MeasureFock() | q[0]
            MeasureFock() | q[1]
        return prog

    def run(self, data: np.ndarray) -> float:
        # Prepare parameter binding for the convolution circuit
        flat = data.reshape(-1)
        param_bind = {}
        for i, val in enumerate(flat):
            param_bind[self.theta[i]] = np.pi if val > self.threshold else 0
        job = qiskit.execute(
            self.circuit,
            self.backend,
            shots=self.conv_shots,
            parameter_binds=[param_bind],
        )
        result = job.result()
        counts = result.get_counts(self.circuit)
        # Compute average probability of measuring |1> for each qubit
        probs = np.zeros(self.n_qubits)
        for bitstring, cnt in counts.items():
            for i, bit in enumerate(bitstring):
                probs[i] += int(bit) * cnt
        probs /= (self.conv_shots * self.n_qubits)
        # Use the first two probabilities as displacements in the photonic circuit
        d1, d2 = probs[0], probs[1]
        # Run the photonic program
        prog = self._build_fraud_program(d1, d2)
        state = self.engine.run(prog).state
        # Expectation value of photon number in both modes
        exp1 = state.expectation_value(Fock(0))
        exp2 = state.expectation_value(Fock(1))
        return float(exp1 + exp2)

__all__ = ["ConvFraudHybridQML"]
