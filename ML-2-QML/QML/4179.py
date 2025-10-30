from __future__ import annotations
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, RY
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit import Aer

# ----------------------------------------------------------------------
# 1. Helper to create a photonic‑style autoencoder circuit
# ----------------------------------------------------------------------
def _autoencoder_circuit(num_latent: int, num_trash: int, input_dim: int) -> QuantumCircuit:
    """
    Builds a quantum autoencoder that encodes `input_dim` classical features
    into `num_latent` latent qubits using a swap‑test style circuit.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encode input features onto the first `input_dim` qubits
    # (here we simply rotate each qubit by a parameterised angle; in practice
    # one would use a more expressive encoder such as a variational layer).
    for i in range(input_dim):
        qc.ry(0.0, qr[i])  # placeholder: parameter will be bound later

    # Entangle latent and trash qubits
    for i in range(num_latent):
        qc.cx(qr[i], qr[i + num_latent])

    # Swap‑test between latent and trash qubits
    for i in range(num_trash):
        qc.cswap(qr[num_latent + i], qr[num_latent + num_trash + i], qr[num_latent + 2 * num_trash])

    # Measurement of an auxiliary qubit to obtain the latent state
    qc.h(qr[-1])
    qc.measure(qr[-1], cr[0])

    return qc

# ----------------------------------------------------------------------
# 2. Helper to create a fraud‑layer circuit (simplified)
# ----------------------------------------------------------------------
def _fraud_layer_circuit(num_qubits: int) -> QuantumCircuit:
    """
    A lightweight parameterised circuit that mimics the photonic fraud‑layer.
    It applies a RealAmplitudes ansatz followed by a measurement of the first
    qubit to produce a scalar expectation.
    """
    qc = QuantumCircuit(num_qubits)
    qc.append(RealAmplitudes(num_qubits, reps=3), range(num_qubits))
    qc.measure_all()
    return qc

# ----------------------------------------------------------------------
# 3. Hybrid fraud detector (quantum implementation)
# ----------------------------------------------------------------------
class HybridFraudDetector:
    """
    Quantum‑style hybrid fraud detector that mirrors the classical
    `HybridFraudDetector`.  It uses a SamplerQNN to obtain a latent vector
    from the autoencoder circuit and a second SamplerQNN to perform the
    fraud‑layer classification.
    """
    def __init__(
        self,
        num_features: int,
        num_latent: int = 3,
        num_trash: int = 2,
        sampler_shots: int = 1024,
    ) -> None:
        self.num_features = num_features
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.sampler_shots = sampler_shots

        # Backend and sampler
        self.backend = Aer.get_backend("qasm_simulator")
        self.sampler = Sampler()

        # Autoencoder circuit and QNN
        self.autoenc_circ = _autoencoder_circuit(num_latent, num_trash, num_features)
        self.autoenc_qnn = SamplerQNN(
            circuit=self.autoenc_circ,
            input_params=[],
            weight_params=self.autoenc_circ.parameters,
            interpret=lambda x: x,  # pass raw counts
            output_shape=(1,),
            sampler=self.sampler,
        )

        # Fraud‑layer circuit and QNN
        self.fraud_circ = _fraud_layer_circuit(num_latent)
        self.fraud_qnn = SamplerQNN(
            circuit=self.fraud_circ,
            input_params=self.fraud_circ.parameters,
            weight_params=[],
            interpret=lambda x: x,
            output_shape=(1,),
            sampler=self.sampler,
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """
        Execute the autoencoder circuit to obtain a latent representation.
        `inputs` is a 1‑D array of classical features.
        """
        # Bind input parameters (here we simply use the feature values as angles)
        param_binds = [{p: val for p, val in zip(self.autoenc_circ.parameters, inputs)}]
        result = self.autoenc_qnn.run(param_binds)
        # The result is a 1‑D array of probabilities; we take the mean as a proxy
        return result.mean(axis=0)

    def classify(self, latent: np.ndarray) -> np.ndarray:
        """
        Run the fraud‑layer circuit on the latent vector and return a scalar
        probability of fraud.
        """
        # Bind latent parameters to the fraud circuit
        param_binds = [{p: val for p, val in zip(self.fraud_circ.parameters, latent)}]
        result = self.fraud_qnn.run(param_binds)
        return result.mean(axis=0)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Full forward pass: encode → classify."""
        latent = self.encode(inputs)
        return self.classify(latent)

__all__ = ["HybridFraudDetector"]
