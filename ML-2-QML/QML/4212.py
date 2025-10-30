"""Quantum version of the hybrid fraud detection model.

The class FraudDetectionHybrid mirrors the classical architecture but replaces
the feature extractor, LSTM, and classifier with quantum circuits:
* StrawberryFields photonic circuit for feature extraction.
* Pennylane variational circuit acting as a quantum LSTM.
* Qiskit sampler circuit for classification.
"""

import torch
import torch.nn as nn
import pennylane as qml
import strawberryfields as sf
from strawberryfields.ops import Dgate, BSgate
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler as QiskitSampler

class FraudDetectionHybrid(nn.Module):
    """Quantum hybrid fraud detection model."""

    def __init__(self,
                 photonic_params,
                 lstm_params,
                 sampler_params):
        super().__init__()
        # Photonic feature extractor
        self.photonic_params = photonic_params
        # Quantum LSTM
        self.lstm_dev = qml.device("default.qubit", wires=4)
        self.lstm_params = nn.Parameter(torch.randn(lstm_params["n_params"]))
        self.lstm_circuit = self._build_lstm_circuit()
        # Sampler classifier
        self.sampler = QiskitSampler()
        self.sampler_params = sampler_params

    def _build_lstm_circuit(self):
        @qml.qnode(self.lstm_dev, interface="torch")
        def circuit(inputs, hidden, params):
            # Encode inputs
            for i, val in enumerate(inputs):
                qml.RX(val, wires=i)
            # Encode hidden state
            for i, val in enumerate(hidden):
                qml.RX(val, wires=i+2)
            # Variational layers
            for i, p in enumerate(params):
                qml.RZ(p, wires=i % 4)
            # Entanglement
            for i in range(3):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(qml.PauliZ(0))
        return circuit

    def _photonic_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Run a StrawberryFields program and return photon‑number expectations."""
        prog = sf.Program(2)
        with prog.context as q:
            # Input displacement
            for i, val in enumerate(x):
                Dgate(val, 0.0) | q[i]
            # Simple beam splitter network
            BSgate(0.5, 0.0) | (q[0], q[1])
        eng = sf.Simulator()
        results = eng.run(prog)
        n0 = eng.state.expectation_value(sf.ops.N(0))
        n1 = eng.state.expectation_value(sf.ops.N(1))
        return torch.tensor([n0, n1], dtype=torch.float32)

    def _sampler_classifier(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """Quantum sampler that outputs a 2‑bit probability vector."""
        qc = QuantumCircuit(1)
        qc.ry(lstm_out, 0)
        qc.measure_all()
        result = self.sampler.run(qc, shots=1024).result()
        counts = result.get_counts()
        p0 = counts.get("0", 0) / 1024
        p1 = counts.get("1", 0) / 1024
        return torch.tensor([p0, p1], dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: shape (batch, seq_len, 2)
        Returns: shape (batch, 2) with fraud probabilities.
        """
        batch, seq_len, _ = inputs.shape
        # Photonic feature extraction
        features = []
        for sample in inputs.reshape(-1, 2):
            feat = self._photonic_feature(sample)
            features.append(feat)
        features = torch.stack(features)  # (batch*seq_len, 2)

        # Quantum LSTM processing
        hidden = torch.zeros(batch * seq_len, 2, device=inputs.device)
        lstm_outputs = []
        for f, h in zip(features, hidden):
            out = self.lstm_circuit(f, h, self.lstm_params)
            lstm_outputs.append(out)
        lstm_outputs = torch.stack(lstm_outputs)  # (batch*seq_len,)

        # Classification via sampler
        probs = []
        for out in lstm_outputs:
            probs.append(self._sampler_classifier(out))
        probs = torch.stack(probs)  # (batch*seq_len, 2)

        # Aggregate over sequence length (e.g., mean)
        probs = probs.view(batch, seq_len, 2).mean(dim=1)
        return probs
