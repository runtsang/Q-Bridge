import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator as StateEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN

class HybridConvEstimator:
    """
    Quantum implementation of the hybrid convolution‑estimator.
    A 2‑qubit (for a 2x2 kernel) circuit encodes the data via
    data‑dependent RX rotations, applies a single variational
    layer of Ry and Rz gates, and measures a Pauli‑Y observable.
    The expectation value of the observable is returned as the
    regression output.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel, implying kernel_size² qubits.
    threshold : float, default 127
        Threshold used to decide the RX rotation angle.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 127):
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.backend = qiskit.Aer.get_backend("qasm_simulator")

        # Data‑encoding parameters
        self.data_params = [Parameter(f"data{i}") for i in range(self.n_qubits)]
        # Variational parameters
        self.var_params = [Parameter(f"theta{i}") for i in range(self.n_qubits)]

        self.circuit = self._build_circuit()
        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=[SparsePauliOp.from_list([("Y" * self.n_qubits, 1)])],
            input_params=self.data_params,
            weight_params=self.var_params,
            estimator=StateEstimator(backend=self.backend),
        )

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        # Data‑encoding: RX rotation with data‑dependent angles
        for i, param in enumerate(self.data_params):
            qc.rx(param, i)
        qc.barrier()
        # Variational layer: one layer of Ry followed by Rz
        for i, param in enumerate(self.var_params):
            qc.ry(param, i)
            qc.rz(param, i)
        qc.barrier()
        qc.measure_all()
        return qc

    def run(self, data: np.ndarray) -> float:
        """
        Execute the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Expectation value of the Pauli‑Y observable across all qubits.
        """
        data_flat = np.reshape(data, (self.n_qubits,))
        input_bind = {
            param: np.pi if val > self.threshold else 0
            for param, val in zip(self.data_params, data_flat)
        }
        result = self.estimator_qnn.run(input_bind)
        return float(result)

__all__ = ["HybridConvEstimator"]
