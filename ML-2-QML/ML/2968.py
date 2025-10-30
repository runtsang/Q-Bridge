import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
import numpy as np
from typing import Tuple, List, Callable, Iterable

class HybridQLSTM(nn.Module):
    """Hybrid LSTM that optionally replaces classical gates with quantum circuits.

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features.
    hidden_dim : int
        Dimensionality of the hidden state.
    n_qubits : int, default=0
        Number of qubits used in the quantum gate submodule. If 0, the layer is fully classical.
    """
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.use_quantum = n_qubits > 0

        out_dim = n_qubits if self.use_quantum else hidden_dim
        self.linear_forget = nn.Linear(input_dim + hidden_dim, out_dim)
        self.linear_input = nn.Linear(input_dim + hidden_dim, out_dim)
        self.linear_update = nn.Linear(input_dim + hidden_dim, out_dim)
        self.linear_output = nn.Linear(input_dim + hidden_dim, out_dim)

        if self.use_quantum:
            self.quantum_gate = self._build_quantum_gate(n_qubits)

    def _build_quantum_gate(self, n_wires: int) -> tq.QuantumModule:
        """Construct a reusable quantum module that implements a small variational circuit."""
        class QGate(tq.QuantumModule):
            def __init__(self, n_wires: int):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [{"input_idx": [i], "func": "rx", "wires": [i]} for i in range(n_wires)]
                )
                self.params = nn.ModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
                self.encoder(qdev, x)
                for wire, gate in enumerate(self.params):
                    gate(qdev, wires=wire)
                for wire in range(self.n_wires):
                    target = 0 if wire == self.n_wires - 1 else wire + 1
                    tqf.cnot(qdev, wires=[wire, target])
                return self.measure(qdev)
        return QGate(n_wires)

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            if self.use_quantum:
                f = torch.sigmoid(self.quantum_gate(self.linear_forget(combined)))
                i = torch.sigmoid(self.quantum_gate(self.linear_input(combined)))
                g = torch.tanh(self.quantum_gate(self.linear_update(combined)))
                o = torch.sigmoid(self.quantum_gate(self.linear_output(combined)))
            else:
                f = torch.sigmoid(self.linear_forget(combined))
                i = torch.sigmoid(self.linear_input(combined))
                g = torch.tanh(self.linear_update(combined))
                o = torch.sigmoid(self.linear_output(combined))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def evaluate(
        self,
        inputs: torch.Tensor,
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute a list of scalar observables for a batch of inputs.

        Parameters
        ----------
        inputs : torch.Tensor
            Input sequence of shape (seq_len, batch, input_dim).
        observables : iterable of callables
            Each callable receives the LSTM output tensor and returns a tensor of scalars.
        shots : int, optional
            If provided, Gaussian noise with variance 1/shots is added.
        seed : int, optional
            Random seed for reproducibility.
        """
        self.eval()
        with torch.no_grad():
            outputs, _ = self.forward(inputs)
            raw = [obs(outputs) for obs in observables]
            scalars = [torch.mean(r, dim=0).cpu().numpy().tolist() for r in raw]
            if shots is not None:
                rng = np.random.default_rng(seed)
                noisy = [[rng.normal(mean, max(1e-6, 1 / shots)) for mean in row] for row in scalars]
                return noisy
            return scalars

class LSTMTagger(nn.Module):
    """Sequence tagging model that uses :class:`HybridQLSTM` as its recurrent core."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = HybridQLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out.view(len(sentence), -1))
        return F.log_softmax(tag_logits, dim=1)

class FastHybridEstimator:
    """Evaluator that runs the hybrid model on a list of inputs and observables, optionally adding shot noise."""
    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def evaluate(
        self,
        inputs: List[torch.Tensor],
        observables: Iterable[Callable[[torch.Tensor], torch.Tensor]],
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        observables = list(observables) or [lambda x: x.mean(dim=-1)]
        results: List[List[float]] = []
        self.model.eval()
        with torch.no_grad():
            for inp in inputs:
                out, _ = self.model(inp)
                row = []
                for obs in observables:
                    val = obs(out)
                    scalar = float(val.mean().cpu())
                    row.append(scalar)
                results.append(row)
        if shots is None:
            return results
        rng = np.random.default_rng(seed)
        noisy = [[rng.normal(mean, max(1e-6, 1 / shots)) for mean in row] for row in results]
        return noisy

__all__ = ["HybridQLSTM", "LSTMTagger", "FastHybridEstimator"]
