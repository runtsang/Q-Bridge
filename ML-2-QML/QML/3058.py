"""
FraudDetectionHybridQNN: Quantum‑classical hybrid model.

This implementation mirrors the classical hybrid above but replaces the
autoencoder with a variational quantum circuit trained via a `SamplerQNN`.
The classifier remains a simple real‑amplitude ansatz that operates on the
latent qubits produced by the autoencoder.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA


class FraudDetectionHybridQNN:
    """
    Quantum hybrid fraud‑detection network.

    Parameters
    ----------
    num_features : int
        Size of the raw transaction vector.
    num_latent : int
        Number of latent qubits produced by the variational autoencoder.
    num_trash : int
        Number of trash qubits used in the swap‑test part of the autoencoder.
    """

    def __init__(
        self,
        num_features: int,
        num_latent: int = 3,
        num_trash: int = 2,
    ) -> None:
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_trash = num_trash
        self._sampler = StatevectorSampler()
        self._circuit = self._build_circuit()
        # All parameters of the circuit are treated as weights
        self.qnn = SamplerQNN(
            circuit=self._circuit,
            input_params=[],
            weight_params=self._circuit.parameters,
            interpret=self._interpret,
            output_shape=1,
            sampler=self._sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        """
        Construct the full autoencoder + classifier circuit.

        The circuit consists of:
          * Basis‑encoding of the input data onto the first `num_features`
            qubits.
          * A RealAmplitudes ansatz that acts on the data + trash qubits.
          * A swap‑test that entangles the trash qubits with the latent
            subspace.
          * Measurement of a single auxiliary qubit to produce a scalar
            output.
        """
        qr = QuantumRegister(
            self.num_features + 2 * self.num_trash + 1, name="q"
        )
        cr = ClassicalRegister(1, name="c")
        qc = QuantumCircuit(qr, cr)

        # Basis encoding – placeholder: user will set X gates externally
        # (see `predict` method).

        # Autoencoder ansatz
        ansatz = RealAmplitudes(
            self.num_features + self.num_trash, reps=5
        )
        qc.compose(
            ansatz,
            range(0, self.num_features + self.num_trash),
            inplace=True,
        )

        # Swap‑test (domain wall + swap)
        aux = self.num_features + self.num_trash
        qc.h(qr[aux])
        for i in range(self.num_trash):
            qc.cswap(qr[aux], qr[self.num_features + i], qr[self.num_features + self.num_trash + i])
        qc.h(qr[aux])

        qc.measure(qr[aux], cr[0])
        return qc

    def _interpret(self, x: np.ndarray) -> float:
        """Return the first element of the state‑vector (Z‑expectation)."""
        return float(x[0])

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Run the QNN on a batch of data.

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, num_features)
            Raw transaction vectors (entries should be 0 or 1 for basis encoding).

        Returns
        -------
        predictions : np.ndarray, shape (n_samples,)
            Estimated fraud scores in the range [0, 1].
        """
        predictions = []
        for sample in data:
            # Copy base circuit to avoid mutating the shared template
            qc = self._circuit.copy()

            # Apply basis encoding: X gate where sample bit is 1
            for i, bit in enumerate(sample):
                if bit > 0.5:
                    qc.x(qc.qubits[i])

            # Execute sampler and compute expectation of Z on the auxiliary qubit
            result = self._sampler.run(qc).result()
            state = result.get_statevector(qc)
            # Compute <Z> = |0><0| - |1><1| expectation
            # Z eigenvalues: +1 for |0>, -1 for |1>
            aux_index = self.num_features + self.num_trash
            prob_0 = np.abs(state[0]) ** 2
            prob_1 = np.abs(state[1]) ** 2
            z_expect = prob_0 - prob_1
            predictions.append((z_expect + 1) / 2)  # map to [0,1]

        return np.array(predictions)

    def fit(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50,
        optimizer: COBYLA | None = None,
    ) -> None:
        """
        Train the variational parameters via a simple COBYLA optimiser.

        Parameters
        ----------
        data : np.ndarray
            Input feature matrix.
        labels : np.ndarray
            Binary fraud labels.
        epochs : int
            Number of optimisation iterations.
        optimizer : COBYLA, optional
            Optimiser instance; defaults to a fresh COBYLA.
        """
        if optimizer is None:
            optimizer = COBYLA()

        def loss_fn(weights):
            # Update qnn weights
            self.qnn.weight_params = weights
            preds = self.predict(data)
            return np.mean((preds - labels) ** 2)

        # Initial guess
        init = np.random.rand(len(self.qnn.weight_params))
        optimizer.optimize(loss=loss_fn, initial_point=init, maxiter=epochs)

__all__ = ["FraudDetectionHybridQNN"]
