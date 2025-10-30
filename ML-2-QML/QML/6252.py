"""ConvEnhancedQML: Variational quantum filter for quanvolution experiments.

The circuit applies a parameterised rotation to each qubit, followed by a
single layer of CNOT entanglement.  The parameters are set to either 0 or π
depending on whether the corresponding pixel value exceeds a learnable
threshold.  The output is the average probability of measuring |1> across
all qubits.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer

class ConvEnhancedQML:
    """
    Variational quantum filter that mirrors the behaviour of ConvEnhanced.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel (kernel_size × kernel_size).
    backend : qiskit.providers.Backend, optional
        Qiskit backend to execute the circuit on.  Defaults to the local
        Aer qasm_simulator.
    shots : int, default 100
        Number of shots per execution.
    threshold : float, default 0.5
        Pixel value threshold used to decide the rotation angle.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend: qiskit.providers.Backend | None = None,
        shots: int = 100,
        threshold: float = 0.5,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size * kernel_size
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        """Builds a parameterised circuit with one rotation per qubit
        and a single entanglement layer."""
        qc = QuantumCircuit(self.n_qubits)
        self.params = [qiskit.circuit.Parameter(f"theta_{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            qc.rx(self.params[i], i)
        # One layer of CNOTs in a linear chain
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data):
        """
        Execute the circuit on classical data.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size) with pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_arr = np.array(data).reshape(1, self.n_qubits)

        param_binds = []
        for dat in data_arr:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.params[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)

        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result()
        counts = result.get_counts(self.circuit)

        total = 0
        for key, val in counts.items():
            ones = sum(int(bit) for bit in key)
            total += ones * val

        return total / (self.shots * self.n_qubits)

__all__ = ["ConvEnhancedQML"]
