from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Pauli
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator

# ----------------------------------------------------------------------
# Fraud‑style parameter set (for consistency with the classical version)
# ----------------------------------------------------------------------
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# Utility to create a fraud‑style layer as a small sub‑circuit
def _fraud_layer_circuit(params: FraudLayerParameters, qubits: Sequence[int]) -> QuantumCircuit:
    qc = QuantumCircuit(len(qubits))
    # Beamsplitter emulation (using CNOT as a placeholder)
    qc.cx(qubits[0], qubits[1])
    # Rotations
    for i, phase in enumerate(params.phases):
        qc.rz(phase, qubits[i])
    # Squeezing (Ry placeholder)
    for i, r in enumerate(params.squeeze_r):
        qc.ry(r, qubits[i])
    # Displacement (Rx placeholder)
    for i, r in enumerate(params.displacement_r):
        qc.rx(r, qubits[i])
    # Kerr (RZZ placeholder)
    for i, k in enumerate(params.kerr):
        qc.rzz(k, qubits[i], qubits[i])
    return qc

# ----------------------------------------------------------------------
# HybridAutoencoder (quantum‑classical)
# ----------------------------------------------------------------------
class HybridAutoencoder:
    """
    Quantum‑classical hybrid autoencoder inspired by the reference pairs.
    The encoder is a variational circuit composed of fraud‑style layers,
    followed by a swap‑test based latent extraction.  The decoder is a
    qiskit EstimatorQNN that maps the latent to a scalar output.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        latent_dim: int = 3,
        num_trash: int = 2,
        fraud_params: Sequence[FraudLayerParameters] | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.num_trash = num_trash

        # Variational ansatz for the encoder
        self.ansatz = RealAmplitudes(num_qubits, reps=5)

        # Fraud‑style layers (optional)
        self.fraud_params = fraud_params or []
        self.fraud_circuits: list[QuantumCircuit] = []
        for params in self.fraud_params:
            self.fraud_circuits.append(_fraud_layer_circuit(params, list(range(num_qubits))))

        # Build swap‑test circuit for latent extraction
        self.latent_circuit = self._build_latent_circuit()

        # Sampler to evaluate the circuit
        self.sampler = StatevectorSampler()

        # Decoder: EstimatorQNN with a single‑qubit circuit
        latent_param = Parameter("latent")
        decoder_circuit = QuantumCircuit(1)
        decoder_circuit.rx(latent_param, 0)
        self.decoder = EstimatorQNN(
            circuit=decoder_circuit,
            observables=Pauli("Z"),
            input_params=[latent_param],
            weight_params=[],
            estimator=StatevectorEstimator(),
        )

    def _build_latent_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.latent_dim + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)
        # Encode part (placeholder: only the ansatz)
        qc.compose(self.ansatz, range(0, self.latent_dim + self.num_trash), inplace=True)
        qc.barrier()
        # Swap test
        aux = self.latent_dim + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.latent_dim + i, self.latent_dim + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    def _encode(self, inputs: np.ndarray) -> np.ndarray:
        # Build the full circuit with input parameters bound to the ansatz
        qc = self.latent_circuit
        param_bindings = {self.ansatz.parameters[i]: inputs[i] for i in range(len(inputs))}
        qc = qc.bind_parameters(param_bindings)

        # Run the circuit and extract the probability of measuring |1> on the aux qubit
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts()
        p1 = sum(cnt for bit, cnt in counts.items() if bit[-1] == "1") / 1024
        return np.array([p1])

    def forward(self, inputs: np.ndarray) -> float:
        latent = self._encode(inputs)
        # Pass the latent through the EstimatorQNN decoder
        return float(self.decoder.forward(torch.tensor(latent)))

__all__ = ["HybridAutoencoder", "FraudLayerParameters"]
