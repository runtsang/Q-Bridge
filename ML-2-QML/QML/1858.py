import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderGen:
    """Quantum autoencoder built with a RealAmplitudes ansatz and a swap‑test reconstruction."""

    def __init__(self, num_latent: int, num_trash: int, input_dim: int, num_qubits: int | None = None):
        algorithm_globals.random_seed = 42
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.input_dim = input_dim
        self.num_qubits = num_qubits or num_latent + 2 * num_trash + 1
        self.circuit = self._build_circuit()
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        ansatz_qr = QuantumRegister(self.num_latent + self.num_trash, "ansatz")
        qc.add_register(ansatz_qr)
        ansatz_circ = RealAmplitudes(self.num_latent + self.num_trash, reps=3)
        qc.compose(ansatz_circ, ansatz_qr, inplace=True)
        aux_qubit = self.num_latent + 2 * self.num_trash
        qc.h(aux_qubit)
        for i in range(self.num_trash):
            qc.cswap(aux_qubit, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux_qubit)
        qc.measure(aux_qubit, cr[0])
        return qc

    def forward(self, params: np.ndarray) -> np.ndarray:
        return self.qnn.forward(params)

def train_qml_autoencoder(
    model: AutoencoderGen,
    data: np.ndarray,
    *,
    epochs: int = 20,
    lr: float = 0.01,
) -> list[float]:
    optimizer = COBYLA()
    history: list[float] = []

    def loss_fn(params):
        loss = 0.0
        for _ in data:
            outputs = model.forward(params)
            loss += (1 - outputs[0]) ** 2
        return loss / len(data)

    for epoch in range(epochs):
        params = optimizer.optimize(num_vars=len(list(model.circuit.parameters)), obj_fun=loss_fn)
        loss = loss_fn(params)
        history.append(loss)
    return history

__all__ = ["AutoencoderGen", "train_qml_autoencoder"]
