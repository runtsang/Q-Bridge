"""Quantum hybrid convolution + autoencoder circuit."""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class ConvGen206:
    """Quantum neural network that fuses a variational convolution filter with a quantum autoencoder."""
    def __init__(
        self,
        kernel_size: int = 2,
        num_latent: int = 3,
        num_trash: int = 2,
        shots: int = 100,
        threshold: float = 127,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.qnn = self._build_qnn(kernel_size, num_latent, num_trash, shots)

    def _build_qnn(self, kernel_size, num_latent, num_trash, shots):
        n_qubits = kernel_size ** 2
        # Convolution filter part
        conv_circuit = QuantumCircuit(n_qubits)
        theta = [Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            conv_circuit.rx(theta[i], i)
        # Add randomness for expressivity
        conv_circuit += qiskit.circuit.random.random_circuit(n_qubits, 2)
        # Quantum autoencoder part
        qr_auto = QuantumRegister(num_latent + 2 * num_trash + 1, "auto")
        cr_auto = ClassicalRegister(1, "c")
        auto_circuit = QuantumCircuit(qr_auto, cr_auto)
        auto_circuit.compose(
            RealAmplitudes(num_latent + num_trash, reps=5),
            range(0, num_latent + num_trash),
            inplace=True,
        )
        auto_circuit.barrier()
        aux = num_latent + 2 * num_trash
        auto_circuit.h(aux)
        for i in range(num_trash):
            auto_circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        auto_circuit.h(aux)
        auto_circuit.measure(aux, cr_auto[0])
        # Combine circuits
        total_circuit = QuantumCircuit(n_qubits + qr_auto.size, ClassicalRegister(1))
        total_circuit.compose(conv_circuit, range(0, n_qubits), inplace=True)
        total_circuit.compose(auto_circuit, range(n_qubits, n_qubits + qr_auto.size), inplace=True)
        # Build QNN
        sampler = Sampler()
        qnn = SamplerQNN(
            circuit=total_circuit,
            input_params=theta,
            weight_params=auto_circuit.parameters,
            interpret=lambda x: x,
            output_shape=1,
            sampler=sampler,
        )
        return qnn

    def run(self, data: np.ndarray) -> float:
        """Run the hybrid QNN on a 2‑D array of shape (kernel_size, kernel_size)."""
        if data.shape!= (self.kernel_size, self.kernel_size):
            raise ValueError("Input data must match the kernel size.")
        angles = [np.pi if val > self.threshold else 0.0 for val in data.flatten()]
        param_binds = {param: angle for param, angle in zip(self.qnn.input_params, angles)}
        result = self.qnn.run(param_binds)
        # The sampler returns a list; take the first element as the expectation
        return result[0].item()

__all__ = ["ConvGen206"]
