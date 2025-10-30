from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_machine_learning.utils import algorithm_globals

def build_quantum_autoencoder(
    latent_dim: int,
    num_trash: int = 2,
    depth: int = 3,
    use_swap_test: bool = True,
) -> SamplerQNN:
    """
    Constructs a variational quantum circuit that encodes a latent vector
    into an entangled state and measures a swap‑test ancilla.
    """
    algorithm_globals.random_seed = 42
    total_qubits = latent_dim + 2 * num_trash + 1
    qr = QuantumRegister(total_qubits, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Variational ansatz on the latent + trash qubits
    ansatz = RealAmplitudes(latent_dim + num_trash, reps=depth)
    qc.append(ansatz, range(0, latent_dim + num_trash))

    # Optional swap‑test on ancilla
    if use_swap_test:
        ancilla = latent_dim + 2 * num_trash
        qc.h(ancilla)
        for i in range(num_trash):
            qc.cswap(ancilla, latent_dim + i, latent_dim + num_trash + i)
        qc.h(ancilla)

    qc.measure(ancilla, cr[0])

    # Interpret the measurement as a real‑valued feature
    def interpret(x: np.ndarray) -> np.ndarray:
        return x.astype(float)

    sampler = Sampler()
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=ansatz.parameters,
        interpret=interpret,
        output_shape=(1,),
        sampler=sampler,
    )
    return qnn

def _params_dict(qnn: SamplerQNN, params: np.ndarray) -> dict:
    """Map a parameter vector to the QNN's weight parameters."""
    return dict(zip(qnn.weight_params, params))

def parameter_shift_gradient(qnn: SamplerQNN, params: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    """
    Estimate the gradient of the QNN output w.r.t. its parameters
    using the parameter‑shift rule.
    """
    grads = np.zeros_like(params)
    for idx in range(len(params)):
        shift = np.array(params, copy=True)
        shift[idx] += epsilon
        out_plus = qnn(_params_dict(qnn, shift)).reshape(-1)
        shift[idx] -= 2 * epsilon
        out_minus = qnn(_params_dict(qnn, shift)).reshape(-1)
        grads[idx] = (out_plus - out_minus) / (2 * epsilon)
    return grads

def train_quantum_latent(
    qnn: SamplerQNN,
    data: np.ndarray,
    epochs: int = 20,
    lr: float = 0.1,
    epsilon: float = 1e-3,
) -> list[float]:
    """
    Simple training loop that optimizes the quantum latent parameters
    using the parameter‑shift gradient and a mean‑squared‑error loss.
    """
    params = np.random.randn(len(qnn.weight_params))
    loss_history: list[float] = []

    for _ in range(epochs):
        preds = np.array([qnn(_params_dict(qnn, params)).reshape(-1) for _ in data])
        loss = ((preds - data) ** 2).mean()
        grads = parameter_shift_gradient(qnn, params, epsilon)
        params -= lr * grads
        loss_history.append(loss)

    return loss_history

__all__ = ["build_quantum_autoencoder", "parameter_shift_gradient", "train_quantum_latent"]
