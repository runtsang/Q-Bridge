"""Quantum‑based hybrid model using variational photonic circuits and a quantum LSTM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields import QuantumInstance

# --------------------------------------------------------------------------- #
# 1. Photonic fraud‑detection parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    sf.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sf.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        sf.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    sf.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sf.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        sf.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        sf.Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

# --------------------------------------------------------------------------- #
# 2. Quantum LSTM block
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell using a small variational circuit per gate."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [0], "func": "rx", "wires": [0]},
                    {"input_idx": [1], "func": "rx", "wires": [1]},
                    {"input_idx": [2], "func": "rx", "wires": [2]},
                    {"input_idx": [3], "func": "rx", "wires": [3]},
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
            for wire in range(self.n_wires):
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        inputs: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:  # type: ignore[override]
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
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

# --------------------------------------------------------------------------- #
# 3. End‑to‑end hybrid quantum model
# --------------------------------------------------------------------------- #
class FraudDetectionHybridQML(nn.Module):
    """Quantum‑based hybrid model: quantum LSTM encoder + photonic fraud detection circuit."""
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        n_qubits: int = 0,
        photonic_params: Iterable[FraudLayerParameters] | None = None,
        backend: str = "gaussian",
        backend_options: dict | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits) if n_qubits > 0 else nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(1, tagset_size)

        # Photonic circuit
        if photonic_params is None:
            raise ValueError("Photonic parameters must be provided.")
        self.photonic_program = build_fraud_detection_program(photonic_params[0], list(photonic_params)[1:])

        # Quantum instance for photonic execution
        if backend == "gaussian":
            backend_obj = sf.backends.GaussianBackend(**(backend_options or {}))
        else:
            backend_obj = sf.backends.FockBackend(**(backend_options or {}))
        self.quantum_instance = QuantumInstance(backend_obj)

    def _run_photonic(self, hidden: torch.Tensor) -> torch.Tensor:
        """Execute the photonic circuit for a batch of hidden vectors."""
        # hidden: [batch_size, 2] – we use the two components as displacements
        batch_size = hidden.shape[0]
        results = []
        for i in range(batch_size):
            prog = sf.Program(2)
            with prog.context as q:
                # Displace modes by hidden vector components
                sf.Dgate(hidden[i, 0].item(), 0.0) | q[0]
                sf.Dgate(hidden[i, 1].item(), 0.0) | q[1]
                # Append the fraud‑detection layers
                for op in self.photonic_program.operations:
                    op.apply(q)
            # Execute on the backend
            exec_res = self.quantum_instance.execute(prog, shots=1)
            # Expectation value of photon number in mode 0 as a simple observable
            exp_val = exec_res.results.expectation_value("n", 0)
            results.append(exp_val)
        return torch.tensor(results, device=hidden.device).unsqueeze(-1)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # Collapse sequence dimension for photonic processing
        lstm_out = lstm_out.view(len(sentence), -1)
        # Run photonic circuit
        photonic_out = self._run_photonic(lstm_out)
        logits = self.hidden2tag(photonic_out)
        return torch.nn.functional.log_softmax(logits, dim=1)

    __all__ = ["FraudDetectionHybridQML", "FraudLayerParameters", "build_fraud_detection_program"]
