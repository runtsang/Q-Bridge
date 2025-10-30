"""Hybrid quantum autoencoder with quantum kernel regularisation.

The module defines `AutoencoderQuantumHybrid` that builds a variational
auto‑encoder circuit using Qiskit.  It mirrors the classical
`AutoencoderHybridNet` but replaces the encoder/decoder with a
parameterised quantum circuit.  A quantum kernel (RBF) is evaluated on
the latent parameters and used as a regulariser during training.

The public API is a factory function `AutoencoderQuantumHybridFactory`
and a training helper `train_qautoencoder_hybrid`.  The implementation
uses Qiskit's simulator back‑ends and the `SamplerQNN` helper from
qiskit‑machine‑learning.
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

# Quantum kernel implementation (adapted from QuantumKernelMethod.py QML)
class QuantumRBFKernelQiskit:
    def __init__(self, n_wires: int = 4) -> None:
        self.n_wires = n_wires
        self.sampler = Sampler()
        self.ansatz = RealAmplitudes(self.n_wires, reps=1)

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        qc = QuantumCircuit(self.n_wires)
        for i, val in enumerate(x):
            qc.ry(val, i)
        for i, val in enumerate(y):
            qc.ry(-val, i)
        result = self.sampler.run(qc).result()
        counts = result.get_counts()
        total = sum(counts.values())
        prob0 = counts.get('0' * self.n_wires, 0) / total
        return prob0

def _autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode latent part with a RealAmplitudes ansatz
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=3),
               range(0, num_latent + num_trash), inplace=True)
    qc.barrier()

    # Swap‑test using an auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

class AutoencoderQuantumHybrid:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        trash_dim: int = 2,
        shots: int = 1024,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim

        self.circuit = _autoencoder_circuit(latent_dim, trash_dim)
        self.weight_params = list(self.circuit.parameters)

        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.weight_params,
            interpret=lambda x: x,
            output_shape=2,
            sampler=Sampler(),
        )
        self.kernel = QuantumRBFKernelQiskit(n_wires=latent_dim)

    def encode(self, data: np.ndarray) -> np.ndarray:
        qc = self.circuit.copy()
        for i, val in enumerate(data):
            qc.ry(val, i)
        result = Sampler().run(qc).result()
        counts = result.get_counts()
        total = sum(counts.values())
        prob1 = counts.get('1', 0) / total
        return np.array([prob1])

    def decode(self, latents: np.ndarray) -> np.ndarray:
        return latents

    def forward(self, data: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(data))

def AutoencoderQuantumHybridFactory(
    input_dim: int,
    *,
    latent_dim: int = 3,
    trash_dim: int = 2,
    shots: int = 1024,
) -> AutoencoderQuantumHybrid:
    return AutoencoderQuantumHybrid(
        input_dim=input_dim,
        latent_dim=latent_dim,
        trash_dim=trash_dim,
        shots=shots,
    )

def train_qautoencoder_hybrid(
    model: AutoencoderQuantumHybrid,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    kernel_weight: float = 0.0,
) -> list[float]:
    opt = COBYLA(maxiter=500)
    history: list[float] = []

    def loss_fn(params: np.ndarray) -> float:
        for param, val in zip(model.weight_params, params):
            param.assign(val)
        recon = np.array([model.forward(d) for d in data])
        mse = np.mean((recon - data)**2)
        if kernel_weight > 0.0:
            latents = np.array([model.encode(d) for d in data])
            k = np.array([[model.kernel.kernel(l1, l2) for l2 in latents] for l1 in latents])
            mse += kernel_weight * (1.0 - k.mean())
        return mse

    initial_params = np.array([p.value for p in model.weight_params], dtype=float)
    for _ in range(epochs):
        result = opt.minimize(loss_fn, initial_params)
        history.append(result.fun)
        initial_params = result.x
    return history

__all__ = [
    "AutoencoderQuantumHybrid",
    "AutoencoderQuantumHybridFactory",
    "train_qautoencoder_hybrid",
]
