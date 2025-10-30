import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

class QuantumAutoencoder:
    """Variational autoencoder built with Qiskit and RealAmplitudes."""
    def __init__(self,
                 num_qubits: int,
                 latent_dim: int,
                 trash_dim: int,
                 reps: int = 5):
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.circuit = self._build_circuit()

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Variational ansatz
        qc.append(RealAmplitudes(self.num_qubits, reps=self.reps), qr)
        # Swap‑test between latent and trash qubits
        aux = self.num_qubits - 1
        qc.h(aux)
        for i in range(self.trash_dim):
            qc.cswap(aux, i, self.latent_dim + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def get_qnn(self) -> SamplerQNN:
        """Wrap the circuit in a Qiskit Machine Learning QNN."""
        sampler = Sampler()
        return SamplerQNN(circuit=self.circuit,
                          input_params=[],
                          weight_params=self.circuit.parameters,
                          sampler=sampler,
                          interpret=lambda x: x)

def build_fraud_detection_circuit(params_list):
    """Create a quantum circuit that mimics the photonic fraud‑detection
    layers using Qiskit gates."""
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    for i, params in enumerate(params_list):
        # Beam splitter analog: RX with theta, RZ with phi
        qc.rx(params.bs_theta, qr[0])
        qc.rx(params.bs_phi, qr[1])
        # Phase gates
        qc.rz(params.phases[0], qr[0])
        qc.rz(params.phases[1], qr[1])
        # Squeezing analog: RZ with squeeze_phi and RX with squeeze_r
        qc.rz(params.squeeze_phi[0], qr[0])
        qc.rx(params.squeeze_r[0], qr[0])
        qc.rz(params.squeeze_phi[1], qr[1])
        qc.rx(params.squeeze_r[1], qr[1])
        # Displacement analog: RZ with displacement_phi and RX with displacement_r
        qc.rz(params.displacement_phi[0], qr[0])
        qc.rx(params.displacement_r[0], qr[0])
        qc.rz(params.displacement_phi[1], qr[1])
        qc.rx(params.displacement_r[1], qr[1])
        # Kerr analog: RZ with kerr[0], RZ with kerr[1]
        qc.rz(params.kerr[0], qr[0])
        qc.rz(params.kerr[1], qr[1])
    qc.measure(qr[0], cr[0])
    return qc
