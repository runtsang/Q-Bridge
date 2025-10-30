from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA

# ----------------------------------------------------------------------
# Imports from the seed modules
# ----------------------------------------------------------------------
from.QCNN import QCNN  # quantum QCNN helper

# ----------------------------------------------------------------------
# Quantum hybrid model
# ----------------------------------------------------------------------
class SharedClassNameQML:
    """
    Quantum counterpart of :class:`SharedClassName`.  Each classical block
    (graph QNN, autoencoder, LSTM, CNN) is replaced by a variational
    circuit that is evaluated with a state‑vector or estimator backend.
    The interface mirrors the classical model to ease side‑by‑side
    experimentation.
    """

    def __init__(
        self,
        graph_arch: list[int],
        autoenc_cfg: dict,
        lstm_cfg: dict,
        cnn_cfg: dict | None = None,
    ) -> None:
        self.graph_arch = graph_arch
        self.autoenc_cfg = autoenc_cfg
        self.lstm_cfg = lstm_cfg
        self.cnn_cfg = cnn_cfg

        self.graph_qnn = self._build_graph_qnn()
        self.autoencoder = self._build_autoencoder()
        self.qlstm = self._build_qlstm()
        self.qcnn = self._build_qcnn()

    # ------------------------------------------------------------------
    # Build a parameterised circuit that mimics the graph architecture
    # ------------------------------------------------------------------
    def _build_graph_qnn(self) -> QuantumCircuit:
        num_qubits = sum(self.graph_arch)
        qc = QuantumCircuit(num_qubits)
        qc.append(RealAmplitudes(num_qubits, reps=3), range(num_qubits))
        return qc

    # ------------------------------------------------------------------
    # Build a simple autoencoder using a sampler QNN
    # ------------------------------------------------------------------
    def _build_autoencoder(self) -> SamplerQNN:
        latent = self.autoenc_cfg["latent_dim"]
        qc = QuantumCircuit(latent)
        qc.append(RealAmplitudes(latent, reps=5), range(latent))
        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=qc,
            input_params=[],
            weight_params=qc.parameters,
            sampler=sampler,
            interpret=lambda x: x,  # identity interpret
        )

    # ------------------------------------------------------------------
    # Build a placeholder LSTM‑style QNN
    # ------------------------------------------------------------------
    def _build_qlstm(self) -> EstimatorQNN:
        hidden = self.lstm_cfg["hidden_dim"]
        qc = QuantumCircuit(hidden)
        estimator = Estimator()
        return EstimatorQNN(
            circuit=qc,
            observables=None,
            input_params=[],
            weight_params=qc.parameters,
            estimator=estimator,
        )

    # ------------------------------------------------------------------
    # Reuse the QCNN helper from the seed
    # ------------------------------------------------------------------
    def _build_qcnn(self) -> EstimatorQNN:
        return QCNN()

    # ------------------------------------------------------------------
    # Forward pass (placeholder: returns statevector of the graph QNN)
    # ------------------------------------------------------------------
    def forward(self, graph_data) -> Statevector:
        """
        `graph_data` is ignored in this simplified example.  The method
        returns the statevector produced by evaluating `self.graph_qnn`
        with a state‑vector backend, which demonstrates the quantum
        side of the pipeline.
        """
        return Statevector(self.graph_qnn)

__all__ = ["SharedClassNameQML"]
