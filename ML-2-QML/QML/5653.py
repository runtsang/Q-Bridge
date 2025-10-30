"""
Quantum‑classical hybrid autoencoder.

The module builds a variational quantum circuit that implements an autoencoder
structure: a classical feature map encodes the input, a RealAmplitudes
ansatz acts as the latent encoder, a swap‑test compares the latent state with
the original, and a second ansatz decodes back to the classical domain.
The entire circuit is wrapped in a :class:`SamplerQNN` for
parameter‑shift training and evaluation.

Typical usage::

    from HybridAutoencoder import QuantumHybridAutoencoder, train_quantum_autoencoder
    qnn = QuantumHybridAutoencoder(input_dim=784, latent_dim=3)
    history = train_quantum_autoencoder(qnn, data, epochs=40)

"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, SwapGate, XGate, HGate
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# --------------------------------------------------------------------------- #
# Quantum autoencoder building blocks
# --------------------------------------------------------------------------- #
def _feature_map(num_qubits: int) -> QuantumCircuit:
    """Classical feature map using RawFeatureVector."""
    return RawFeatureVector(num_qubits)

def _latent_ansatz(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Variational ansatz for the latent space."""
    return RealAmplitudes(num_qubits, reps=reps)

def _decode_ansatz(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """Variational ansatz for decoding back to classical domain."""
    return RealAmplitudes(num_qubits, reps=reps)

def _swap_test(circuit: QuantumCircuit, control: int, target: int) -> None:
    """Insert a swap‑test into the circuit."""
    circuit.h(control)
    circuit.cswap(control, target, target + 1)
    circuit.h(control)

# --------------------------------------------------------------------------- #
# Main wrapper
# --------------------------------------------------------------------------- #
class QuantumHybridAutoencoder:
    """
    A SamplerQNN that implements a quantum autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the classical input.
    latent_dim : int
        Number of qubits used for the latent representation.
    """
    def __init__(self, input_dim: int, latent_dim: int = 3) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_qubits = latent_dim
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.sampler = StatevectorSampler()

        # Build the full circuit
        self.circuit = self._build_circuit()

        # Define the QNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # raw probability amplitudes
            output_shape=(self.input_dim,),
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """Construct the full variational circuit."""
        qr = QuantumRegister(self.num_qubits, "q")
        cr = ClassicalRegister(self.num_qubits, "c")
        circuit = QuantumCircuit(qr, cr)

        # Feature map (classical encoding)
        fm = _feature_map(self.num_qubits)
        circuit.compose(fm, inplace=True)

        # Latent encoder
        latent = _latent_ansatz(self.num_qubits)
        circuit.compose(latent, inplace=True)

        # Swap test with a copy of the original state
        # (for simplicity we reuse the same qubits)
        _swap_test(circuit, 0, 1)

        # Decoder
        decode = _decode_ansatz(self.num_qubits)
        circuit.compose(decode, inplace=True)

        # Measurement
        circuit.measure(qr, cr)
        return circuit

    def forward(self, params: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """
        Evaluate the QNN for given parameters and inputs.

        Parameters
        ----------
        params : np.ndarray
            Flattened variational parameters.
        inputs : np.ndarray
            Batch of classical data of shape (batch, input_dim).

        Returns
        -------
        outputs : np.ndarray
            Reconstructed data of shape (batch, input_dim).
        """
        # qiskit expects a list of input parameter values; in this example
        # we ignore classical inputs and rely on the feature map.
        return self.qnn.forward(params, inputs).reshape(-1, self.input_dim)

# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #
def train_quantum_autoencoder(
    qnn: QuantumHybridAutoencoder,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.01,
    optimizer_cls: type = COBYLA,
    verbose: bool = False,
) -> list[float]:
    """
    Train the quantum autoencoder using a classical optimizer.

    Parameters
    ----------
    qnn : QuantumHybridAutoencoder
        The QNN to train.
    data : np.ndarray
        Training data of shape (N, input_dim).
    epochs : int
        Number of optimization epochs.
    lr : float
        Learning rate for the optimizer.
    optimizer_cls : type
        Optimizer class from qiskit_machine_learning.optimizers.
    verbose : bool
        Print loss per epoch.

    Returns
    -------
    history : list[float]
        Reconstruction loss per epoch.
    """
    opt = optimizer_cls(maxiter=200)
    params = np.random.rand(len(list(qnn.circuit.parameters)))  # random init
    history: list[float] = []

    for epoch in range(epochs):
        def cost(p):
            preds = qnn.forward(p, data)
            return np.mean((preds - data) ** 2)

        params = opt.minimize(cost, params, options={"ftol": 1e-6, "xtol": 1e-6})
        loss = cost(params)
        history.append(loss)
        if verbose:
            print(f"Epoch {epoch+1:3d} | loss = {loss:.6f}")
    return history

__all__ = ["QuantumHybridAutoencoder", "train_quantum_autoencoder"]
