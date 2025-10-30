import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector, random
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler as StatevectorSampler

class ConvGen189QML:
    """Quantum‑centric counterpart of ConvGen189.  It implements a
    parameterised convolutional filter (QuanvCircuit), a simple quantum
    fully‑connected layer using random gates, and a SamplerQNN that
    produces a 2‑class probability vector.  The circuit is built on
    Qiskit Aer and can be executed on a simulator or a real device.
    """

    def __init__(self,
                 kernel_size: int = 2,
                 qc_shots: int = 1024,
                 sampler_shots: int = 1024,
                 threshold: int = 127):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.backend = AerSimulator()
        self.qc_shots = qc_shots
        self.sampler_shots = sampler_shots
        self._build_conv_circuit()
        self._build_qfc_circuit()
        self._build_sampler()

    def _build_conv_circuit(self) -> None:
        self.conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector("theta", self.n_qubits)
        self.conv_circuit.rx(self.theta, range(self.n_qubits))
        self.conv_circuit.barrier()
        self.conv_circuit += random.random_circuit(self.n_qubits, 2)
        self.conv_circuit.measure_all()

    def _build_qfc_circuit(self) -> None:
        self.qfc_circuit = QuantumCircuit(4)
        self.qfc_circuit.h(range(4))
        self.qfc_circuit.cx(0, 1)
        self.qfc_circuit.cx(2, 3)
        self.qfc_circuit.rx(0.5, 0)
        self.qfc_circuit.cx(0, 2)
        self.qfc_circuit.measure_all()

    def _build_sampler(self) -> None:
        inputs = ParameterVector("input", 2)
        weights = ParameterVector("weight", 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        self.sampler = SamplerQNN(circuit=qc,
                                  input_params=inputs,
                                  weight_params=weights,
                                  sampler=StatevectorSampler())

    def run_conv(self, data: np.ndarray) -> float:
        """Execute the convolutional filter on a 2×2 patch."""
        flat = data.reshape(self.n_qubits)
        param_binds = [{self.theta[i]: np.pi if val > self.threshold else 0}
                       for i, val in enumerate(flat)]
        job = self.backend.run(self.conv_circuit,
                               shots=self.qc_shots,
                               parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self.conv_circuit)
        total = sum(counts.values())
        ones = sum(int(bit) * cnt for bit, cnt in counts.items() for _ in range(1))
        return ones / (total * self.n_qubits)

    def run_qfc(self, inputs: np.ndarray) -> np.ndarray:
        """Quantum fully‑connected layer that maps a 4‑dim vector to probabilities."""
        job = self.backend.run(self.qfc_circuit, shots=self.qc_shots)
        result = job.result()
        counts = result.get_counts(self.qfc_circuit)
        probs = np.array([counts.get(f"{i:04b}", 0) for i in range(16)], dtype=float)
        probs /= probs.sum()
        return probs

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """Return a 2‑class probability vector from the SamplerQNN."""
        return self.sampler(inputs)

__all__ = ["ConvGen189QML"]
