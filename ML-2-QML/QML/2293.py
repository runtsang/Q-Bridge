"""Hybrid quantum autoencoder and estimator."""

from __future__ import annotations

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp


class HybridAutoencoderEstimator:
    """Quantum hybrid architecture combining a variational autoencoder and a parameterized estimator."""
    def __init__(self, num_latent: int = 3, num_trash: int = 2) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.autoencoder_circuit = self._build_autoencoder_circuit()
        self.estimator_circuit = self._build_estimator_circuit()
        self.autoencoder_qnn = self._build_autoencoder_qnn()
        self.estimator_qnn = self._build_estimator_qnn()

    def _build_autoencoder_circuit(self) -> QuantumCircuit:
        """Construct the variational autoencoder circuit with a swap‑test."""
        qr = QuantumRegister(self.num_latent + 2 * self.num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        # Ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=5)
        circuit.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)
        circuit.barrier()
        # swap‑test
        aux = self.num_latent + 2 * self.num_trash
        circuit.h(aux)
        for i in range(self.num_trash):
            circuit.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    def _build_estimator_circuit(self) -> QuantumCircuit:
        """Simple 1‑qubit parameterized estimator with a Y‑observable."""
        p_in = Parameter("in")
        p_w = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(p_in, 0)
        qc.rx(p_w, 0)
        return qc

    def _build_autoencoder_qnn(self) -> SamplerQNN:
        """Wrap the autoencoder circuit in a SamplerQNN."""
        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.autoencoder_circuit,
            input_params=[],
            weight_params=self.autoencoder_circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=sampler,
        )

    def _build_estimator_qnn(self) -> EstimatorQNN:
        """Wrap the estimator circuit in an EstimatorQNN."""
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = StatevectorEstimator()
        return EstimatorQNN(
            circuit=self.estimator_circuit,
            observables=observable,
            input_params=[self.estimator_circuit.parameters[0]],
            weight_params=[self.estimator_circuit.parameters[1]],
            estimator=estimator,
        )

    def predict(self, latent: list[float], weight: float) -> float:
        """Run the estimator QNN for a single latent sample."""
        params = {
            self.estimator_circuit.parameters[0]: latent[0],
            self.estimator_circuit.parameters[1]: weight,
        }
        return self.estimator_qnn.predict(params)

    def encode(self, data: list[float]) -> list[float]:
        """Encode classical data into a latent vector via the autoencoder QNN."""
        # For demonstration, we call the QNN without explicit parameters.
        return self.autoencoder_qnn.predict({})

    def __repr__(self) -> str:
        return f"<HybridAutoencoderEstimator latent={self.num_latent} trash={self.num_trash}>"

__all__ = ["HybridAutoencoderEstimator"]
