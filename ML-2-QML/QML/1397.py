import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.circuit.library import TwoLocal
from typing import Optional

class ConvGen361:
    """
    Variational quantum filter that mimics a convolutional kernel.
    The circuit consists of a trainable parameterised rotation layer followed
    by a 2‑local entangling block.  Parameters can be initialised from a
    classical kernel if supplied.
    """
    def __init__(
        self,
        kernel_size: int = 2,
        backend: Optional[qiskit.providers.BaseBackend] = None,
        shots: int = 1024,
        threshold: float = 0.5,
        init_from_kernel: Optional[np.ndarray] = None,
    ):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.threshold = threshold
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Parameterised rotation layer
        self.theta = [Parameter(f"θ{i}") for i in range(self.n_qubits)]
        self.circuit = QuantumCircuit(self.n_qubits)
        for i, p in enumerate(self.theta):
            self.circuit.rx(p, i)
        self.circuit.barrier()

        # 2‑local entangling block
        entanglement = TwoLocal(
            self.n_qubits,
            rotation="ry",
            entanglement="cz",
            reps=1,
            insert_barriers=False,
        )
        self.circuit.append(entanglement.to_instruction(), range(self.n_qubits))
        self.circuit.measure_all()

        # Initialise parameters from classical kernel if provided
        if init_from_kernel is not None:
            flat = init_from_kernel.flatten()
            for i, val in enumerate(flat):
                self.circuit.assign_parameters({self.theta[i]: val}, inplace=True)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on the provided 2‑D data array.
        Data values are thresholded to set the rotation angles.
        Returns the average probability of measuring |1> across all qubits.
        """
        flat = data.flatten()
        param_bindings = [
            {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(flat)}
        ]
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_bindings,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        # Compute average |1> probability
        total_ones = 0
        for bitstring, n in counts.items():
            ones = bitstring.count("1")
            total_ones += ones * n
        prob = total_ones / (self.shots * self.n_qubits)
        return prob
