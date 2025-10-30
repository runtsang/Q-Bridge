import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# Classical convolutional filter (from Conv.py)
class ConvFilter:
    def __init__(self, kernel_size=2, threshold=0.0):
        self.kernel_size = kernel_size
        self.threshold = threshold

    def run(self, data):
        data = np.reshape(data, (1, self.kernel_size ** 2))
        return np.mean(data > self.threshold).astype(float)

class AutoencoderHybrid:
    def __init__(self, num_latent=3, num_trash=2, shots=100, backend=None):
        algorithm_globals.random_seed = 42
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler(self.backend)
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.shots = shots
        self.conv = ConvFilter(kernel_size=2, threshold=0.0)
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self):
        num_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz
        qc.append(RealAmplitudes(num_qubits, reps=5), qr)

        # Swap test
        aux = num_qubits - 1
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        return qc

    def run(self, data):
        # data shape (samples, features)
        conv_features = np.array([self.conv.run(d) for d in data])
        # Bind parameters to circuit (one parameter per sample for illustration)
        param_binds = [{self.circuit.parameters[i % len(self.circuit.parameters)]: conv_features[i]} for i in range(len(data))]
        result = self.sampler.run(self.circuit, shots=self.shots, parameter_binds=param_binds)
        counts = result.get_counts()
        total = 0
        for key, val in counts.items():
            ones = sum(int(b) for b in key)
            total += ones * val
        return total / (self.shots * self.circuit.num_qubits)

    def train(self, data, epochs=100):
        opt = COBYLA(maxfun=1000)
        def loss_fn(params):
            param_binds = [{p: v} for p, v in zip(self.circuit.parameters, params)]
            result = self.sampler.run(self.circuit, shots=self.shots, parameter_binds=param_binds)
            counts = result.get_counts()
            recon = np.array([sum(int(b) for b in k) for k in counts.keys()]) / self.shots
            return np.mean((recon - data)**2)
        initial_params = np.random.rand(len(self.circuit.parameters))
        opt.optimize(epochs, loss_fn, initial_params)
        return opt.x

__all__ = ["AutoencoderHybrid"]
