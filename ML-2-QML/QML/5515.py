"""Hybrid quantum autoencoder.

The quantum part implements a swap‑test based autoencoder as in the
original QML seed.  It can be used standalone or as the latent
representation in the classical hybrid model.
"""

from __future__ import annotations

import numpy as np
import qiskit as qk
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class QuantumAutoencoder:
    """Swap‑test based quantum autoencoder with optional QCNN feature map.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits that encode the compressed state.
    num_trash : int
        Number of auxiliary qubits used for the swap test.
    use_qcnn : bool
        If True, prepend a QCNN feature map before the ansatz.
    """
    def __init__(self, num_latent: int = 3, num_trash: int = 2, use_qcnn: bool = False):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.use_qcnn = use_qcnn
        self.sampler = Sampler()
        self._build_circuit()

    def _qc_nn_circuit(self, params: ParameterVector) -> QuantumCircuit:
        """QCNN feature map circuit for 8‑dimensional data."""
        qc = QuantumCircuit(8)
        # Simple 8‑qubit feature map using rotations
        for i in range(8):
            qc.ry(params[i], i)
        return qc

    def _ansatz(self, num_qubits: int) -> QuantumCircuit:
        return RealAmplitudes(num_qubits, reps=5)

    def _auto_encoder_circuit(self, num_latent: int, num_trash: int) -> QuantumCircuit:
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode part
        circuit.compose(self._ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()

        # Swap test
        auxiliary = num_latent + 2 * num_trash
        circuit.h(auxiliary)
        for i in range(num_trash):
            circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary)
        circuit.measure(auxiliary, cr[0])
        return circuit

    def _build_circuit(self):
        ae_circ = self._auto_encoder_circuit(self.num_latent, self.num_trash)
        if self.use_qcnn:
            # Build a parameter vector for the QCNN feature map
            qc_params = ParameterVector("θ", 8)
            qc = QuantumCircuit(8 + ae_circ.num_qubits)
            qc.compose(self._qc_nn_circuit(qc_params), range(8))
            qc.compose(ae_circ, range(8, 8 + ae_circ.num_qubits))
            self.circuit = qc
            self.input_params = qc_params
        else:
            self.circuit = ae_circ
            self.input_params = []

    def get_qnn(self) -> SamplerQNN:
        """Return a SamplerQNN instance that can be used as a layer."""
        return SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def train(self, training_data: np.ndarray, *, epochs: int = 100, lr: float = 0.1):
        """Train the quantum autoencoder using COBYLA."""
        algorithm_globals.random_seed = 42
        qnn = self.get_qnn()
        opt = COBYLA(maxiter=epochs)

        def loss_fn(params: np.ndarray) -> float:
            qnn.set_weights(params)
            # No classical inputs: we still need to pass an empty array
            probs = qnn.forward(np.zeros((1, 0)))
            # measurement outcome 1 corresponds to reconstruction success
            return -probs[0][1]  # maximize probability of 1

        opt.minimize(loss_fn, np.zeros(len(self.circuit.parameters)))
        return opt.x

__all__ = ["QuantumAutoencoder"]
