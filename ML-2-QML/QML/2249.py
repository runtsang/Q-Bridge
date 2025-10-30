"""Hybrid quantum-classical estimator with a quantum autoencoder.

This module defines EstimatorAutoencoder that uses a variational
quantum circuit to encode the input, compress into a latent subspace,
and then decodes to produce a scalar prediction.  The circuit is
implemented with Qiskit and trained using a COBYLA optimiser.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import Sampler as SamplerPrimitive
from qiskit.utils import algorithm_globals

class EstimatorAutoencoder:
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        num_trash: int = 2,
        reps: int = 5,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_trash = num_trash
        self.reps = reps
        algorithm_globals.random_seed = seed
        self.sampler = SamplerPrimitive()

        # Define input parameters
        self.input_params = [Parameter(f"x_{i}") for i in range(input_dim)]

        # Build the circuit
        self.circuit = self._build_circuit()

        # Extract weight parameters (exclude input params)
        self.weight_params = [p for p in self.circuit.parameters if p not in self.input_params]

        # Create the SamplerQNN
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            interpret=lambda x: x[0],
            output_shape=1,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode input features into the first latent_dim qubits
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)

        # Trainable ansatz on latent qubits
        ansatz = RealAmplitudes(self.latent_dim, reps=self.reps)
        qc.append(ansatz, range(self.latent_dim))

        # Swap test for compression
        qc.barrier()
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        # Decoder ansatz
        decoder = RealAmplitudes(self.latent_dim, reps=self.reps)
        qc.append(decoder, range(self.latent_dim))

        # Measurement for output
        qc.measure(0, cr[0])

        return qc

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        preds = []
        for x in X:
            param_bind = {p: val for p, val in zip(self.input_params, x)}
            result = self.sampler.run(self.circuit, parameter_binds=[param_bind]).result()
            counts = result.get_counts()
            total = sum(counts.values())
            p1 = counts.get("1", 0) / total if total > 0 else 0.0
            preds.append(p1)
        return np.array(preds)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        maxiter: int = 100,
        tol: float = 1e-3,
    ) -> dict:
        def loss_fn(params):
            self.circuit.assign_parameters(params, inplace=True)
            preds = []
            for x in X:
                param_bind = {p: val for p, val in zip(self.input_params, x)}
                result = self.sampler.run(self.circuit, parameter_binds=[param_bind]).result()
                counts = result.get_counts()
                total = sum(counts.values())
                p1 = counts.get("1", 0) / total if total > 0 else 0.0
                preds.append(p1)
            preds = np.array(preds)
            return np.mean((preds - y) ** 2)

        optimizer = COBYLA(maxiter=maxiter, tol=tol)
        init_params = np.random.rand(len(self.weight_params))
        opt_res = optimizer.optimize(
            num_vars=len(self.weight_params),
            objective_function=loss_fn,
            initial_point=init_params,
        )
        self.circuit.assign_parameters(opt_res[0], inplace=True)
        return {"params": opt_res[0], "fun": opt_res[1]}

__all__ = ["EstimatorAutoencoder"]
