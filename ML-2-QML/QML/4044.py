import numpy as np
import qiskit
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit.random import random_circuit
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler as Sampler

class HybridFCL:
    """
    Quantum analogue of HybridFCL that implements a convolution gate,
    a single‑qubit fully‑connected rotation and a quantum auto‑encoder
    using a RealAmplitudes ansatz and a swap‑test decoder.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_threshold: float = 0.0,
                 fc_n_features: int = 1,
                 latent_dim: int = 3,
                 num_trash: int = 2,
                 shots: int = 100):
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        # Convolution layer
        n_qubits = conv_kernel ** 2
        self.conv_circuit = qiskit.QuantumCircuit(n_qubits)
        self.conv_params = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            self.conv_circuit.rx(self.conv_params[i], i)
        self.conv_circuit.barrier()
        self.conv_circuit += random_circuit(n_qubits, 2)
        self.conv_circuit.measure_all()
        self.conv_threshold = conv_threshold
        # Fully‑connected layer
        self.fc_circuit = qiskit.QuantumCircuit(1)
        self.fc_param = qiskit.circuit.Parameter("theta_fc")
        self.fc_circuit.ry(self.fc_param, 0)
        self.fc_circuit.measure_all()
        # Auto‑encoder circuit
        def ae_ansatz(num_latent, num_trash):
            qr = qiskit.QuantumRegister(num_latent + 2 * num_trash + 1, "q")
            cr = qiskit.ClassicalRegister(1, "c")
            circuit = qiskit.QuantumCircuit(qr, cr)
            circuit.compose(RealAmplitudes(num_latent + num_trash, reps=5), range(0, num_latent + num_trash), inplace=True)
            circuit.barrier()
            aux = num_latent + 2 * num_trash
            circuit.h(aux)
            for i in range(num_trash):
                circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
            circuit.h(aux)
            circuit.measure(aux, cr[0])
            return circuit
        self.ae_circuit = ae_ansatz(latent_dim, num_trash)
        sampler = Sampler()
        self.ae_qnn = SamplerQNN(
            circuit=self.ae_circuit,
            input_params=[],
            weight_params=self.ae_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )

    def conv_run(self, data: np.ndarray) -> float:
        """
        Execute the quantum convolution and return the average probability
        of measuring |1> over all qubits.
        """
        data_flat = data.reshape(-1)
        binds = [{p: np.pi if val > self.conv_threshold else 0} for val, p in zip(data_flat, self.conv_params)]
        job = qiskit.execute(self.conv_circuit, self.backend, shots=self.shots, parameter_binds=binds)
        result = job.result().get_counts(self.conv_circuit)
        ones = sum(int(k.count("1")) * v for k, v in result.items())
        return ones / (self.shots * len(self.conv_params))

    def fc_run(self, theta: float) -> float:
        """
        Run the fully‑connected rotation and return the expectation of Z.
        """
        job = qiskit.execute(self.fc_circuit, self.backend, shots=self.shots,
                             parameter_binds=[{self.fc_param: theta}])
        result = job.result().get_counts(self.fc_circuit)
        exp = (result.get("0",0) - result.get("1",0)) / self.shots
        return exp

    def ae_run(self, theta: float) -> float:
        """
        Map the scalar theta into the weight parameters of the auto‑encoder
        and return the reconstructed probability from the auxiliary qubit.
        """
        params = {p: theta for p in self.ae_circuit.parameters}
        output = self.ae_qnn.forward(params)
        return output[1].item()

    def run(self, data: np.ndarray):
        """
        Execute the full quantum pipeline.

        Parameters
        ----------
        data : np.ndarray
            2‑D array of shape (conv_kernel, conv_kernel).

        Returns
        -------
        Tuple[float, float, float]
            (conv_out, fc_out, recon_out)
        """
        conv_out = self.conv_run(data)
        fc_out = self.fc_run(conv_out)
        recon = self.ae_run(fc_out)
        return conv_out, fc_out, recon
