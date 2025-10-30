import json
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2) -> SamplerQNN:
    """Construct a variational autoencoder circuit that can be used as a quantum latent encoder."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    def ansatz(nq: int):
        return RealAmplitudes(nq, reps=5)

    def auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    qc = auto_encoder_circuit(num_latent, num_trash)

    def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
        for i in range(a, b):
            circuit.x(i)
        return circuit

    dw_circ = domain_wall(QuantumCircuit(num_latent + 2 * num_trash + 1), 0, num_latent + 2 * num_trash + 1)
    qc.compose(dw_circ, inplace=True)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        return x

    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

def train_quantum_autoencoder(
    qnn: SamplerQNN,
    training_data: np.ndarray,
    *,
    num_iterations: int = 100,
    learning_rate: float = 0.1,
    optimizer_cls=COBYLA,
    device: str = "statevector",
) -> list[float]:
    """Simple training loop for the quantum autoencoder using classical optimizers."""
    opt = optimizer_cls(maxiter=num_iterations, tol=1e-6)
    history = []

    def loss_fn(params):
        qnn.set_weights(params)
        outputs = qnn.forward(training_data)
        # Reconstruction loss; here we use mean‑squared difference between
        # expectation values and the original data.
        recon = outputs
        loss = np.mean((recon - training_data) ** 2)
        return loss

    params0 = np.array(qnn.get_weights(), dtype=np.float64)
    result = opt.minimize(loss_fn, params0)
    history.append(result.fun)
    return history
