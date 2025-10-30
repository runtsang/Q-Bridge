import numpy as np
from qiskit import QuantumCircuit, Parameter
from qiskit.circuit.library import RealAmplitudes

class HybridEstimator:
    """Quantum component library for the hybrid estimator."""
    def __init__(self):
        self.regression_circuit, self.reg_params = self._regression_circuit()
        self.fcl_circuit, self.fcl_params = self._fcl_circuit()
        self.conv_circuit, self.conv_params = self._conv_circuit()
        self.autoencoder_circuit = self._autoencoder_circuit()

    def _regression_circuit(self):
        params = [Parameter(f"p{i}") for i in range(3)]
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(params[0], 0)
        qc.rx(params[1], 0)
        qc.rz(params[2], 0)
        return qc, params

    def _fcl_circuit(self, n_qubits: int = 4):
        qc = QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i, t in enumerate(theta):
            qc.ry(t, i)
        qc.barrier()
        qc += RealAmplitudes(n_qubits, reps=2)
        return qc, theta

    def _conv_circuit(self, kernel_size: int = 2):
        n_qubits = kernel_size ** 2
        qc = QuantumCircuit(n_qubits)
        theta = [Parameter(f"t{i}") for i in range(n_qubits)]
        for i, t in enumerate(theta):
            qc.rx(t, i)
        qc.barrier()
        qc += RealAmplitudes(n_qubits, reps=1)
        return qc, theta

    def _autoencoder_circuit(self, num_latent: int = 3, num_trash: int = 2):
        n_qubits = num_latent + 2 * num_trash + 1
        qc = QuantumCircuit(n_qubits)
        for i in range(num_latent + num_trash):
            qc.rx(Parameter(f"enc{i}"), i)
        qc.h(num_latent + 2 * num_trash)
        for i in range(num_trash):
            qc.cswap(num_latent + 2 * num_trash, num_latent + i, num_latent + num_trash + i)
        qc.h(num_latent + 2 * num_trash)
        qc.measure(num_latent + 2 * num_trash, 0)
        return qc

__all__ = ["HybridEstimator"]
