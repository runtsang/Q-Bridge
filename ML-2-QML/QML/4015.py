"""
UnifiedQSampler: quantum sampler for the latent space.
"""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RawFeatureVector
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
import numpy as np
from typing import Callable, List

def make_latent_sampler_circuit(num_qubits: int, reps: int = 3) -> QuantumCircuit:
    """
    Builds a parameterised circuit that encodes a latent vector into amplitudes.
    """
    qr = QuantumRegister(num_qubits, "q")
    cr = ClassicalRegister(num_qubits, "c")
    qc = QuantumCircuit(qr, cr)
    # Feature map: RawFeatureVector for each input qubit
    feature_map = RawFeatureVector(num_qubits)
    qc.append(feature_map, qr)
    # Variational ansatz
    var_ansatz = RealAmplitudes(num_qubits, reps=reps)
    qc.append(var_ansatz, qr)
    # Measurement
    qc.measure(qr, cr)
    return qc

class UnifiedQSampler:
    """
    Wraps a Qiskit SamplerQNN around the latent sampler circuit.
    """
    def __init__(self,
                 num_qubits: int,
                 reps: int = 3,
                 input_params: List = None,
                 weight_params: List = None,
                 interpret: Callable = None):
        self.circuit = make_latent_sampler_circuit(num_qubits, reps)
        self.sampler = Sampler()
        # If no parameters given, create default
        input_params = input_params or []
        weight_params = weight_params or list(self.circuit.parameters)
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=input_params,
            weight_params=weight_params,
            interpret=interpret or (lambda x: x),
            output_shape=(len(input_params),),
            sampler=self.sampler,
        )

    def sample(self, params: np.ndarray, shots: int = 1024) -> np.ndarray:
        """
        Evaluate the QNN with given variational parameters.
        """
        return self.qnn.predict(params, shots=shots)

    def optimize(self,
                 target: np.ndarray,
                 initial: np.ndarray,
                 maxiter: int = 200,
                 tol: float = 1e-3) -> np.ndarray:
        """
        Optimize the variational parameters to match a target distribution
        using the COBYLA optimizer.
        """
        def loss_func(params):
            samples = self.sample(params, shots=5000)
            # Convert samples to probability estimate
            probs = np.bincount(samples.astype(int), minlength=2**len(params)) / samples.size
            return np.linalg.norm(probs - target)

        optimizer = COBYLA(maxiter=maxiter, tol=tol)
        return optimizer.optimize(initial, loss_func)

__all__ = ["UnifiedQSampler", "make_latent_sampler_circuit"]
