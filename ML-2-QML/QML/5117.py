import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf

from.QLSTM import QLSTM as QuantumQLSTM
from.QTransformerTorch import TransformerBlockQuantum
from.SamplerQNN import SamplerQNN as QuantumSamplerQNN
from.FastBaseEstimator import FastBaseEstimator


class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced hybrid model that mirrors the classical HybridQLSTM
    but replaces LSTM gates and optional transformer blocks with
    quantum modules.  The interface remains identical, enabling
    side‑by‑side comparison or combined training.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        transformer_blocks: int = 0,
        transformer_heads: int = 1,
        transformer_ffn: int = 128,
        use_sampler: bool = False,
        q_device: tq.QuantumDevice | None = None,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional quantum transformer encoder
        if transformer_blocks > 0:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockQuantum(
                        embedding_dim,
                        transformer_heads,
                        transformer_ffn,
                        n_qubits,
                        n_qubits,
                        1,
                        q_device=q_device,
                    )
                    for _ in range(transformer_blocks)
                ]
            )
        else:
            self.transformer = None

        # Quantum LSTM gates or classical fallback
        if n_qubits > 0:
            self.lstm = QuantumQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = QuantumSamplerQNN()
        else:
            self.sampler = None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing log‑softmax tag probabilities.
        """
        embeds = self.word_embeddings(sentence)
        if self.transformer is not None:
            embeds = self.transformer(embeds)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(logits, dim=1)

    def evaluate(
        self,
        observables: list,
        parameter_sets: list,
    ) -> list:
        """
        Evaluate observables using the classical FastBaseEstimator
        on the model's output.  Quantum observables are not directly
        supported in this simplified wrapper.
        """
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Sample from the model's output distribution using the quantum
        SamplerQNN.  The input tensor is a batch of sentences.
        """
        if not self.sampler:
            raise RuntimeError("Sampler not enabled for this model.")
        probs = self.forward(inputs).exp()
        flat_probs = probs.reshape(-1, probs.size(-1))
        return self.sampler(flat_probs)
