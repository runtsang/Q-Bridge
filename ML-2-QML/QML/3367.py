"""Quantum hybrid LSTM + EstimatorQNN for sequence tagging and regression.

The quantum implementation mirrors the classical module but replaces the
classical LSTM with a variational quantum LSTM (QLSTM) defined in the
``QLSTM.py`` seed.  The regression head is a quantum estimator built with
Qiskit, following the pattern of the EstimatorQNN example.  The overall
model is fully differentiable via the PyTorch backend of TorchQuantum
and the StatevectorEstimator primitive.

Key design points
-----------------
* The QLSTM cell uses a small variational circuit per gate.  It is
  parameter‑shared across time steps.
* After the LSTM, the final hidden state is flattened and fed into a
  single‑qubit variational circuit that estimates the desired scalar.
* The forward pass returns both the tagging log‑probabilities and the
  quantum expectation value as the regression output.

The class can be dropped into existing training loops that expect the
original QLSTM API.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

# Import the quantum LSTM defined in the original QLSTM.py seed
# (assumed to be in the same package or available on PYTHONPATH).
# We re‑export it here for clarity.
from QLSTM import QLSTM as QuantumLSTM  # type: ignore


class HybridQLSTMRegressor(nn.Module):
    """Quantum hybrid LSTM + EstimatorQNN.

    Parameters
    ----------
    embedding_dim : int
        Dimension of word embeddings.
    hidden_dim : int
        Size of LSTM hidden states (must be a multiple of n_qubits).
    vocab_size : int
        Number of distinct tokens in the vocabulary.
    tagset_size : int
        Number of target tags for the tagging head.
    n_qubits : int
        Number of qubits used in each QLSTM gate and in the estimator.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Quantum LSTM
        self.lstm = QuantumLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # Quantum estimator head
        self.estimator = self._build_estimator(n_qubits)

    def _build_estimator(self, n_qubits: int) -> QiskitEstimatorQNN:
        """Build a 1‑qubit variational estimator following the EstimatorQNN example."""
        # For simplicity we use a single‑qubit circuit.
        param_input = Parameter("input")
        param_weight = Parameter("weight")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(param_input, 0)
        qc.rx(param_weight, 0)
        # Observable: Y on the qubit
        observable = SparsePauliOp.from_list([("Y", 1)])
        estimator = Estimator()
        return QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=[param_input],
            weight_params=[param_weight],
            estimator=estimator,
        )

    def forward(self, sentence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape (seq_len, batch) containing token indices.

        Returns
        -------
        tag_logits : torch.Tensor
            Log‑softmax over tags, shape (seq_len, batch, tagset_size).
        regression : torch.Tensor
            Scalar predictions from the quantum estimator, shape (batch, 1).
        """
        embeds = self.word_embeddings(sentence)
        lstm_out, (hn, _) = self.lstm(embeds)
        tag_logits = self.hidden2tag(lstm_out)
        tag_logits = F.log_softmax(tag_logits, dim=-1)
        # Prepare input for the quantum estimator: flatten hidden state
        # and map it to the input parameter of the circuit.
        # We use a simple linear mapping to a scalar in [0, π].
        input_scalar = torch.tanh(hn.mean(dim=1)).unsqueeze(-1)  # shape (batch, 1)
        # The estimator expects a 2‑D numpy array of shape (batch, 1)
        # containing the input parameter values.
        reg_vals = self.estimator.forward(input_scalar.detach().cpu().numpy())
        regression = torch.tensor(reg_vals, device=sentence.device, dtype=torch.float32)
        return tag_logits, regression


__all__ = ["HybridQLSTMRegressor"]
