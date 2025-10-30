"""Quantum autoencoder using a swap‑test circuit and a RealAmplitudes ansatz."""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.utils import algorithm_globals

def _domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Inject a domain wall by X‑gates on a contiguous block of qubits."""
    for q in range(start, end):
        circuit.x(q)
    return circuit

def _auto_encoder_circuit(
    num_latent: int,
    num_trash: int,
    depth: int = 5,
    domain_wall: bool = True,
) -> QuantumCircuit:
    """Build the swap‑test based auto‑encoder circuit."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz on latent + trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=depth)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Optional domain wall injection
    if domain_wall:
        circuit = _domain_wall(circuit, 0, num_latent + num_trash)

    # Swap‑test
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit

def _interpret(x):
    """Interpret the sampler output as the probability of measuring 0."""
    return x[0]  # probability of |0⟩ on the auxiliary qubit

class QuantumAutoencoder:
    """Quantum auto‑encoder wrapper returning a SamplerQNN."""

    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        depth: int = 5,
        domain_wall: bool = True,
    ) -> None:
        self.circuit = _auto_encoder_circuit(
            num_latent=num_latent,
            num_trash=num_trash,
            depth=depth,
            domain_wall=domain_wall,
        )
        self.sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=_interpret,
            output_shape=2,
            sampler=self.sampler,
        )

    def __call__(self, *args, **kwargs):
        return self.qnn(*args, **kwargs)

    def train(
        self,
        training_data: np.ndarray,
        *,
        epochs: int = 50,
        lr: float = 0.01,
        optimizer_cls=algorithm_globals.get_optimizer,
    ) -> list[float]:
        """Simple training loop for the quantum auto‑encoder."""
        opt = optimizer_cls(name="COBYLA", options={"maxiter": 200})
        history = []

        for epoch in range(epochs):
            # For simplicity we only support a single batch equal to the data size
            preds = self.qnn(input_data=training_data)
            loss = np.mean((preds - training_data) ** 2)
            opt.minimize(lambda p: np.mean((self.qnn(p, input_data=training_data) - training_data) ** 2),
                         p0=np.random.rand(len(self.circuit.parameters)))
            history.append(loss)

        return history

__all__ = ["QuantumAutoencoder"]
