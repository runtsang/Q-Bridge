"""Quantum‑enhanced LSTM tagger with fast expectation evaluation.

Implements the same public API as the classical :class:`QLSTMTagger`
but realises the LSTM gates as small variational circuits.
The module contains a lightweight estimator that can evaluate
expectation values of user‑supplied observables for a batch of
parameter sets, optionally using shot‑level sampling.

The implementation follows the structure of the original
``QLSTM.py`` but expands the quantum layer to use
parameterised rotation and entanglement gates, and
provides a convenient :class:`FastBaseEstimator` that
mirrors the behaviour of the FastEstimator from the
reference pair.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from collections.abc import Iterable, Sequence
from typing import Callable, List, Tuple

ScalarObservable = Callable[[torch.Tensor], torch.Tensor | float]

# ----------------------------------------------------------------------
#  FastBaseEstimator – evaluates expectation values for a parametric circuit
# ----------------------------------------------------------------------
class FastBaseEstimator:
    """
    Lightweight estimator that evaluates expectation values of
    Pauli observables for a parametrised quantum circuit.

    Parameters
    ----------
    circuit : tq.QuantumModule
        A torchquantum module that can be called with a batch of parameters.
    """

    def __init__(self, circuit: tq.QuantumModule) -> None:
        self.circuit = circuit
        self.param_names = list(circuit.parameters())

    def evaluate(
        self,
        observables: Iterable[torch.Tensor],
        parameter_sets: Sequence[Sequence[float]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """
        Compute expectation values for each parameter set.

        Parameters
        ----------
        observables : Iterable[torch.Tensor]
            Each observable should be a Pauli operator matrix
            (e.g. ``tq.PauliZ``).
        parameter_sets : Sequence[Sequence[float]]
            Each inner sequence must match ``len(self.param_names)``.
        shots : int | None, default None
            If given, perform Monte‑Carlo sampling using the provided
            number of shots.  Otherwise compute the exact expectation
            value using state vector simulation.
        seed : int | None, default None
            Random seed for reproducibility of Monte‑Carlo sampling.

        Returns
        -------
        List[List[float]]
            Outer list corresponds to parameter sets,
            inner list to observables.
        """
        results: List[List[float]] = []
        for values in parameter_sets:
            # Bind parameters to the circuit
            self.circuit.set_parameters(values)
            if shots is None:
                # Exact expectation
                exp = []
                for obs in observables:
                    exp.append(float(self.circuit.expectation_value(obs)))
                results.append(exp)
            else:
                # Monte‑Carlo sampling
                rng = np.random.default_rng(seed)
                samples = self.circuit.sample(shots, seed=seed)
                exp = []
                for obs in observables:
                    # obs is a Pauli matrix; evaluate expectation per sample
                    exp.append(float(samples.expectation_value(obs)))
                results.append(exp)
        return results

# ----------------------------------------------------------------------
#  Quantum LSTM Layer
# ----------------------------------------------------------------------
class _QLayer(tq.QuantumModule):
    """
    Variational circuit that implements a single LSTM gate.
    The circuit consists of a trainable layer of Rx/Ry rotations
    followed by a chain of CNOTs to entangle the qubits.
    """

    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Parameterised rotations
        self.rxs = nn.ModuleList([tq.RX(has_params=True) for _ in range(n_wires)])
        self.rys = nn.ModuleList([tq.RY(has_params=True) for _ in range(n_wires)])
        # Entangling layer
        self.cnot_chain = [
            tqf.cnot if i == n_wires - 1 else lambda dev, w=i: tqf.cnot(dev, wires=[w, w + 1])
            for i in range(n_wires)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gate circuit.

        Parameters
        ----------
        x : torch.Tensor
            Batch of real parameters of shape ``(batch, n_wires)``.
            These are interpreted as angles for the initial rotation layer.

        Returns
        -------
        torch.Tensor
            Qubit measurement outcomes in {0,1} for each wire.
        """
        dev = tq.QuantumDevice(
            n_wires=self.n_wires,
            bsz=x.shape[0],
            device=x.device,
        )
        for i, (rx, ry) in enumerate(zip(self.rxs, self.rys)):
            rx(dev, wires=i, params=x[:, i].unsqueeze(-1))
            ry(dev, wires=i, params=x[:, i].unsqueeze(-1))
        for i in range(self.n_wires):
            self.cnot_chain[i](dev, wires=[i, (i + 1) % self.n_wires])
        return tq.MeasureAll(tq.PauliZ)(dev)

# ----------------------------------------------------------------------
#  Quantum LSTM Tagger
# ----------------------------------------------------------------------
class QLSTMTagger(nn.Module):
    """
    Quantum‑enhanced LSTM tagger with a parameter‑sweep evaluation
    interface analogous to the classical :class:`QLSTMTagger`.

    Parameters
    ----------
    embedding_dim : int
    hidden_dim : int
    vocab_size : int
    tagset_size : int
    n_qubits : int
        Number of qubits used in each variational gate.
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
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.n_qubits = n_qubits

        # Quantum gates
        self.forget = _QLayer(n_qubits)
        self.input = _QLayer(n_qubits)
        self.update = _QLayer(n_qubits)
        self.output = _QLayer(n_qubits)

        # Linear maps from classical features to angles
        self.linear_forget = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(embedding_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(embedding_dim + hidden_dim, n_qubits)

        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single sentence.

        Parameters
        ----------
        sentence : torch.Tensor
            LongTensor of shape ``(seq_len,)``.
        """
        embeds = self.word_embeddings(sentence).unsqueeze(0)  # (1, seq_len, embed)
        batch_size, seq_len, _ = embeds.shape
        hx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=embeds.device)

        outputs = []
        for t in range(seq_len):
            x_t = embeds[0, t, :]
            combined = torch.cat([x_t, hx.squeeze(0)], dim=0).unsqueeze(0)  # (1, hidden+embed)
            f = torch.sigmoid(
                self.forget(self.linear_forget(combined))
            )
            i = torch.sigmoid(
                self.input(self.linear_input(combined))
            )
            g = torch.tanh(
                self.update(self.linear_update(combined))
            )
            o = torch.sigmoid(
                self.output(self.linear_output(combined))
            )
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        lstm_out = torch.cat(outputs, dim=0)  # (seq_len, hidden)
        tag_logits = self.hidden2tag(lstm_out)
        return F.log_softmax(tag_logits, dim=1)

    # ------------------------------------------------------------------
    #  Evaluation utilities (FastBaseEstimator inspired)
    # ------------------------------------------------------------------
    def _apply_noise(
        self,
        logits: torch.Tensor,
        shots: int | None = None,
        seed: int | None = None,
    ) -> torch.Tensor:
        if shots is None:
            return logits
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 1.0 / np.sqrt(shots), size=logits.shape)
        return logits + torch.from_numpy(noise).to(logits.device)

    def evaluate(
        self,
        sentences: Sequence[torch.Tensor],
        shots: int | None = None,
        seed: int | None = None,
        observables: Iterable[ScalarObservable] | None = None,
    ) -> List[List[float]]:
        """
        Evaluate over sentences with optional shot‑level noise.

        Parameters
        ----------
        sentences : Sequence[torch.Tensor]
            Iterable of token index tensors.
        shots : int | None, default None
            If set, Gaussian noise is added to logits to emulate
            shot‑level sampling.
        seed : int | None, default None
            Random seed for reproducibility.
        observables : Iterable[ScalarObservable] | None, default None
            Functions mapping logits to scalars.  If omitted, mean
            of logits is returned.

        Returns
        -------
        List[List[float]]
            Outer list: sentences, inner list: observables.
        """
        results: List[List[float]] = []
        self.eval()
        with torch.no_grad():
            for sentence in sentences:
                logits = self.forward(sentence)
                logits = self._apply_noise(logits, shots, seed)

                if observables is None:
                    obs_values = [float(logits.mean().item())]
                else:
                    obs_values = [
                        float(obs(logits).mean().item()) if isinstance(obs(logits), torch.Tensor)
                        else float(obs(logits))
                        for obs in observables
                    ]
                results.append(obs_values)
        return results

__all__ = ["QLSTMTagger", "FastBaseEstimator"]
