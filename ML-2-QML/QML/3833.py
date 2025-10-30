from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit as QiskitCircuit
from qiskit import assemble, transpile
from qiskit.circuit import Parameter
from qiskit.providers.aer import AerSimulator

class QuantumExpectationCircuit:
    """
    Two‑qubit parameterised circuit that returns the expectation value of Z⊗Z.
    """

    def __init__(self, backend: AerSimulator, shots: int = 1024) -> None:
        self.backend = backend
        self.shots = shots
        self.circuit = QiskitCircuit(2)
        self.theta = Parameter("theta")
        self.circuit.h([0, 1])
        self.circuit.barrier()
        self.circuit.ry(self.theta, 0)
        self.circuit.ry(self.theta, 1)
        self.circuit.measure_all()

    def _expectation_from_counts(self, counts) -> float:
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()], dtype=int)
        return np.sum(states * probs)

    def run(self, params: np.ndarray) -> np.ndarray:
        expectations = []
        for param in params:
            compiled = transpile(self.circuit, self.backend)
            qobj = assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: param}],
            )
            job = self.backend.run(qobj)
            result = job.result().get_counts()
            expectations.append(self._expectation_from_counts(result))
        return np.array(expectations)

class HybridFunction(torch.autograd.Function):
    """
    Differentiable wrapper that feeds a scalar through the quantum circuit.
    """

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        circuit: QuantumExpectationCircuit,
        shift: float = np.pi / 2,
    ) -> torch.Tensor:
        ctx.shift = shift
        ctx.circuit = circuit
        params = input_tensor.detach().cpu().numpy()
        expectation = circuit.run(params)
        result = torch.tensor(expectation, dtype=input_tensor.dtype, device=input_tensor.device)
        ctx.save_for_backward(input_tensor, result)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        input_tensor, _ = ctx.saved_tensors
        shift = ctx.shift
        grad_input = torch.zeros_like(input_tensor)
        input_cpu = input_tensor.detach().cpu().numpy()
        for i, val in enumerate(input_cpu):
            right = ctx.circuit.run([val + shift])
            left = ctx.circuit.run([val - shift])
            grad_input[i] = (right - left) / 2.0
        return grad_input.to(input_tensor.device) * grad_output, None, None

class QLSTMCell(tq.QuantumModule):
    """
    Quantum LSTM cell where each gate is a small quantum circuit.
    """

    class QGate(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                ]
            )
            self.params = nn.ModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.measure = tq.MeasureAll(tq.PauliZ)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=x.shape[0], device=x.device)
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_wires: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires

        self.forget = self.QGate(n_wires)
        self.input = self.QGate(n_wires)
        self.update = self.QGate(n_wires)
        self.output = self.QGate(n_wires)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_wires)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_wires)

    def forward(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(
        self,
        inputs: torch.Tensor,
        states: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridSequenceClassifier(nn.Module):
    """
    Hybrid quantum‑classical sequence classifier.
    Embedding → Quantum LSTM → Quantum expectation head → Sigmoid.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        n_qubits: int,
        num_classes: int = 2,
        shift: float = np.pi / 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.qlstm = QLSTMCell(embedding_dim, hidden_dim, n_qubits)
        backend = AerSimulator()
        self.quantum_head = QuantumExpectationCircuit(backend)
        self.shift = shift
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        input_ids : torch.LongTensor, shape (batch, seq_len)

        Returns
        -------
        torch.Tensor, shape (batch, num_classes)
            Probabilities for each class.
        """
        embedded = self.embedding(input_ids)                    # (batch, seq_len, embed_dim)
        qlstm_out, _ = self.qlstm(embedded.transpose(0, 1))     # (seq_len, batch, hidden_dim)
        last_hidden = qlstm_out[-1]                              # (batch, hidden_dim)
        quantum_output = HybridFunction.apply(
            last_hidden[:, 0], self.quantum_head, self.shift
        )                                                     # (batch,)
        logits = quantum_output.unsqueeze(-1)                   # (batch, 1)
        probs = self.sigmoid(logits)                            # (batch, 1)
        return probs

__all__ = ["HybridSequenceClassifier"]
