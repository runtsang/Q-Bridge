"""Variational autoencoder using Qiskit and a swap‑test similarity measure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, Optional

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate
from qiskit.quantum_info import Statevector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA, SPSA
from qiskit.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class QAutoencoderConfig:
    """Parameters for the quantum autoencoder circuit."""
    num_latent: int = 3
    num_trash: int = 2
    reps: int = 5
    optimizer: str = "COBYLA"  # or "SPSA"
    max_iter: int = 100
    seed: int = 42


# --------------------------------------------------------------------------- #
# Circuit construction
# --------------------------------------------------------------------------- #

def _build_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    """Return a RealAmplitudes ansatz with *reps* repetitions."""
    return RealAmplitudes(num_qubits, reps=reps)


def _swap_test(circuit: QuantumCircuit, q0: int, q1: int) -> None:
    """Insert a swap‑test between qubits *q0* and *q1*."""
    circuit.h(q0)
    circuit.barrier()
    circuit.cswap(q0, q1, q1)
    circuit.h(q0)


def autoencoder_circuit(cfg: QAutoencoderConfig) -> QuantumCircuit:
    """Full encoder circuit with an embedded swap‑test."""
    n = cfg.num_latent + 2 * cfg.num_trash + 1  # +1 auxiliary qubit
    qr = QuantumRegister(n, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode the data with a RealAmplitudes ansatz
    ansatz = _build_ansatz(cfg.num_latent + cfg.num_trash, cfg.reps)
    qc.compose(ansatz, range(0, cfg.num_latent + cfg.num_trash), inplace=True)

    # Swap‑test between latent and trash
    aux = cfg.num_latent + 2 * cfg.num_trash
    _swap_test(qc, aux, cfg.num_latent)  # compare first latent with first trash
    qc.measure(aux, cr[0])

    return qc


# --------------------------------------------------------------------------- #
# QNN wrapper
# --------------------------------------------------------------------------- #

def build_qnn(cfg: QAutoencoderConfig) -> SamplerQNN:
    """Return a SamplerQNN that maps classical inputs to a latent vector."""
    qc = autoencoder_circuit(cfg)
    sampler = Sampler()
    # No classical inputs – the circuit is fully parameterised
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=lambda x: x,  # raw measurement probabilities
        output_shape=(2,),
        sampler=sampler,
    )
    return qnn


# --------------------------------------------------------------------------- #
# Training routine
# --------------------------------------------------------------------------- #

def train_qautoencoder(
    cfg: QAutoencoderConfig,
    training_data: Iterable[float],
    *,
    verbose: bool = False,
) -> tuple[SamplerQNN, Sequence[float]]:
    """Train the quantum autoencoder on *training_data*."""
    algorithm_globals.random_seed = cfg.seed
    qnn = build_qnn(cfg)

    # Define the loss: mean squared error between input and reconstructed state
    def loss_fn(params: np.ndarray) -> float:
        qnn.set_parameters(params)
        # Sample from the circuit
        result = qnn.forward([])
        probs = result[0]
        # Ideal state is |0⟩, so target probability vector is [1, 0]
        target = np.array([1.0, 0.0])
        return float(np.mean((probs - target) ** 2))

    # Optimizer selection
    if cfg.optimizer.upper() == "COBYLA":
        opt = COBYLA(maxiter=cfg.max_iter)
    elif cfg.optimizer.upper() == "SPSA":
        opt = SPSA(maxiter=cfg.max_iter)
    else:
        raise ValueError(f"Unsupported optimizer {cfg.optimizer}")

    # Initial parameters
    init_params = np.random.rand(len(qnn.weights))

    # Run optimisation
    history: list[float] = []
    opt_result = opt.minimize(loss_fn, init_params, show_output=verbose)
    for h in opt_result["fun_history"]:
        history.append(float(h))

    # Apply the final parameters
    qnn.set_parameters(opt_result["x"])
    return qnn, history


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def plot_loss(history: Sequence[float]) -> None:
    """Simple loss curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(history, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("Quantum autoencoder training")
    plt.legend()
    plt.tight_layout()
    plt.show()


__all__ = [
    "QAutoencoderConfig",
    "autoencoder_circuit",
    "build_qnn",
    "train_qautoencoder",
    "plot_loss",
]
