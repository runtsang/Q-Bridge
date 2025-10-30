import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

# Helper: build a parameterised self‑attention block
def _build_self_attention_circuit(num_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    rotx = [Parameter(f'rotx_{i}') for i in range(num_qubits)]
    roty = [Parameter(f'roty_{i}') for i in range(num_qubits)]
    rotz = [Parameter(f'rotz_{i}') for i in range(num_qubits)]
    ent  = [Parameter(f'ent_{i}') for i in range(num_qubits-1)]
    for i in range(num_qubits):
        qc.rx(rotx[i], i)
        qc.ry(roty[i], i)
        qc.rz(rotz[i], i)
    for i in range(num_qubits-1):
        qc.crx(ent[i], i, i+1)
    return qc

# Quantum auto‑encoder that embeds the self‑attention block
def QuantumAutoEncoder(num_latent: int, num_trash: int) -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    # Ansatz for the latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    # Self‑attention subcircuit on the same qubits
    sa_circuit = _build_self_attention_circuit(num_latent + num_trash)
    # Combine
    qc = ansatz.compose(sa_circuit, inplace=True)
    qc.measure_all()
    # Interpret the output as a simple vector
    def identity_interpret(x): return x
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# Hybrid network that exposes the quantum auto‑encoder as a PyTorch‑like module
class SelfAttentionAutoEncoderAttentionNet:
    """A thin wrapper that mimics a nn.Module interface around a quantum auto‑encoder.

    The ``forward`` method simply evaluates the SamplerQNN on a dummy input,
    returning the two‑dimensional output produced by the quantum circuit.
    """

    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        self.qnn = QuantumAutoEncoder(num_latent, num_trash)

    def forward(self, inputs: np.ndarray | None = None) -> np.ndarray:
        # The original quantum auto‑encoder has no input parameters;
        # we simply evaluate the circuit once and return the result.
        return self.qnn.forward([])

__all__ = ["SelfAttentionAutoEncoderAttentionNet"]
