import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridFunction:
    """Differentiable quantum expectation layer."""
    def __init__(self, circuit: QuantumCircuit, shots: int = 1024) -> None:
        self.circuit = circuit
        self.backend = Aer.get_backend('qasm_simulator')
        self.shots = shots
    def __call__(self, params: list[float]):
        circ = self.circuit.bind_parameters({p: v for p, v in zip(self.circuit.parameters, params)})
        job = execute(circ, self.backend, shots=self.shots)
        counts = job.result().get_counts()
        prob0 = counts.get('0', 0) / self.shots
        return prob0

class QuantumHybridAutoencoder:
    """Quantum autoencoder that blends a RealAmplitudes ansatz, swap‑test reconstruction,
    and a sampler‑style QNN decoder."""
    def __init__(self, input_dim: int, latent_dim: int = 3,
                 num_trash: int = 2, shots: int = 1024) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.shots = shots
        self.backend = Aer.get_backend('qasm_simulator')
        # Encoding ansatz
        self.ansatz = RealAmplitudes(latent_dim + num_trash, reps=5)
        # Sampler QNN decoder
        inputs = ParameterVector('input', 2)
        weights = ParameterVector('weight', 4)
        qc = QuantumCircuit(2)
        qc.ry(inputs[0], 0)
        qc.ry(inputs[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[0], 0)
        qc.ry(weights[1], 1)
        qc.cx(0, 1)
        qc.ry(weights[2], 0)
        qc.ry(weights[3], 1)
        sampler = StatevectorSampler(Aer.get_backend('aer_simulator'))
        self.decoder_qnn = SamplerQNN(circuit=qc, input_params=inputs,
                                      weight_params=weights, sampler=sampler)
        # Hybrid expectation layer
        self.hybrid = HybridFunction(self.ansatz, shots=self.shots)
    def _domain_wall(self, circuit: QuantumCircuit, indices: list[int]) -> QuantumCircuit:
        for i in indices:
            circuit.x(i)
        return circuit
    def encode(self, data: np.ndarray):
        circuits = []
        for sample in data:
            qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1)
            cr = ClassicalRegister(1)
            qc = QuantumCircuit(qr, cr)
            # Encode first (latent+trash) bits as X gates
            for i, val in enumerate(sample[:self.latent_dim + 2 * self.num_trash]):
                if val >= 0.5:
                    qc.x(i)
            qc.compose(self.ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)
            aux = self.latent_dim + 2 * self.num_trash
            qc.h(aux)
            for i in range(self.num_trash):
                qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
            qc.h(aux)
            qc.measure(aux, cr[0])
            circuits.append(qc)
        return circuits
    def decode(self, circuits):
        reconstructions = []
        for qc in circuits:
            job = execute(qc, self.backend, shots=self.shots)
            counts = job.result().get_counts()
            prob0 = counts.get('0', 0) / self.shots
            reconstructions.append(prob0)
        recon_arr = np.array(reconstructions).reshape(-1, 1)
        # Wrap scalar into 2‑dim vector for the sampler QNN
        qnn_input = np.hstack([recon_arr, np.ones_like(recon_arr)])
        return self.decoder_qnn(qnn_input)
    def forward(self, data: np.ndarray):
        circuits = self.encode(data)
        return self.decode(circuits)
