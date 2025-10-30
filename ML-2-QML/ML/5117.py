import torch
import torch.nn as nn
import torch.nn.functional as F

from.QLSTM import QLSTM as ClassicalQLSTM
from.QTransformerTorch import TransformerBlockClassical
from.SamplerQNN import SamplerQNN as ClassicalSamplerQNN
from.FastBaseEstimator import FastBaseEstimator


class HybridQLSTM(nn.Module):
    """
    Hybrid classical model that optionally augments the original QLSTM with a
    transformer encoder and a lightweight sampler.  All components are
    drop‑in replacements for the original QLSTM API, making it trivial to
    switch between pure classical, quantum, or hybrid modes.
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
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Optional transformer encoder to enrich contextual representations
        if transformer_blocks > 0:
            self.transformer = nn.Sequential(
                *[
                    TransformerBlockClassical(
                        embedding_dim, transformer_heads, transformer_ffn
                    )
                    for _ in range(transformer_blocks)
                ]
            )
        else:
            self.transformer = None

        # Classical or quantum LSTM gate implementation
        if n_qubits > 0:
            self.lstm = ClassicalQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

        # Optional sampler for probabilistic output generation
        self.use_sampler = use_sampler
        if use_sampler:
            self.sampler = ClassicalSamplerQNN()
        else:
            self.sampler = None

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning log‑softmax tag probabilities.
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
        Evaluate a list of scalar observables over several parameter sets
        using the classical FastBaseEstimator.  Observables are callables
        that accept the model output tensor and return a scalar.
        """
        estimator = FastBaseEstimator(self)
        return estimator.evaluate(observables, parameter_sets)

    def sample(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Generate a sample from the model's output distribution using the
        embedded SamplerQNN.  The input tensor is assumed to be a batch
        of sentences; only the first dimension is used for sampling.
        """
        if not self.sampler:
            raise RuntimeError("Sampler not enabled for this model.")
        probs = self.forward(inputs).exp()
        # The sampler expects a 2‑dim tensor; flatten the batch if needed
        flat_probs = probs.reshape(-1, probs.size(-1))
        return self.sampler(flat_probs)
