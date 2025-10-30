"""
HybridQLSTM – quantum‑enhanced implementation of the hybrid architecture.
Each sub‑component is a quantum circuit or photonic program, evaluated via
simulation.  The public API matches the classical version, enabling
side‑by‑side experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Iterable, Tuple, List


# --------------------------------------------------------------------------- #
# Photonic fraud‑detection parameters
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _apply_layer(
    modes: Iterable[object], params: FraudLayerParameters, *, clip: bool
) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Creates a Strawberry Fields program that mirrors the photonic fraud‑detection
    architecture.  The program is evaluated by the Gaussian state engine."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


# --------------------------------------------------------------------------- #
# Classical‑style convolution replaced by a quantum filter
# --------------------------------------------------------------------------- #
class Conv:
    """Quantum convolution filter implemented with a parameter‑ized Qiskit circuit."""

    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 100, threshold: float = 127):
        self.n_qubits = kernel_size ** 2
        self._circuit = QuantumCircuit(self.n_qubits)
        self.theta = ParameterVector(f"theta", self.n_qubits)
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += QuantumCircuit(self.n_qubits).random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: torch.Tensor) -> torch.Tensor:
        """Run the quantum filter on a 2‑D kernel.  Returns the mean |1> probability."""
        kernel = data.view(-1, self.n_qubits)
        param_binds = []
        for row in kernel:
            bind = {self.theta[i]: torch.pi if val > self.threshold else 0 for i, val in enumerate(row)}
            param_binds.append(bind)
        job = self.backend.run(self._circuit, parameters=param_binds, shots=self.shots)
        result = job.result().get_counts(self._circuit)
        probs = torch.tensor([sum(int(bit) for bit in key) / self.n_qubits for key, val in result.items()])
        return probs.mean()


# --------------------------------------------------------------------------- #
# Quantum LSTM cell – gates realised by small quantum circuits
# --------------------------------------------------------------------------- #
class QLSTM(nn.Module):
    """Quantum‑enhanced LSTM cell with parameter‑shared quantum modules."""

    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int) -> None:
            super().__init__()
            self.n_wires = n_wires
            self.encoder = tq.GeneralEncoder(
                [
                    {"input_idx": [i], "func": "rx", "wires": [i]}
                    for i in range(n_wires)
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
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0) -> None:
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

    def forward(self, inputs: torch.Tensor, states: tuple[torch.Tensor, torch.Tensor] | None = None):
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
        return torch.cat(outputs, dim=0), (hx, cx)

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


# --------------------------------------------------------------------------- #
# Quantum classifier – variational ansatz with measurement observables
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Creates a layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]
    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Hybrid model – quantum implementation
# --------------------------------------------------------------------------- #
class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced hybrid architecture.  The forward pass mirrors the classical
    implementation but replaces every sub‑module with its quantum counterpart.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        num_qubits: int,
        depth: int,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: List[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # Quantum convolution filter
        self.conv = Conv(kernel_size=2, threshold=127)

        # Quantum LSTM cell
        self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=num_qubits)

        # Quantum classifier
        self.classifier_circuit, self.enc, self.wt, self.obs = build_classifier_circuit(num_qubits, depth)
        self.classifier_params = self.classifier_circuit.parameters
        self.classifier_engine = tq.QuantumEngine(
            backend="aer_simulator",
            device="cpu",
            bsz=1,
            n_wires=num_qubits,
        )
        self.classifier_engine.add_circuit(self.classifier_circuit)

        # Photonic fraud‑detection program
        self.fraud_prog = build_fraud_detection_program(fraud_input_params, fraud_layers)
        self.fraud_engine = sf.Engine("gaussian_state")

    def forward(self, sentence: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sentence : LongTensor (seq_len, batch)
            Token indices.
        features : FloatTensor (batch, num_qubits)
            Features that will be processed by the quantum classifier.

        Returns
        -------
        Tensor
            Fraud‑detection probability of shape (batch, 1).
        """
        # 1. Embedding
        embeds = self.word_embeddings(sentence)  # (seq_len, batch, embed)

        # 2. Quantum LSTM
        lstm_out, _ = self.lstm(embeds)  # (seq_len, batch, hidden)

        # 3. Quantum classifier – use last hidden state as measurement input
        cls_input = lstm_out[-1]  # (batch, hidden)

        # Bind classifier parameters
        binding = {p: cls_input.squeeze().cpu().numpy() for p in self.enc}
        self.classifier_engine.bind_parameters(binding)
        result = self.classifier_engine.run()
        logits = torch.tensor(result["expectation"][-1], device=features.device).unsqueeze(-1)

        # 4. Photonic fraud detection
        prog = self.fraud_prog
        prog.bind_parameters({k: v for k, v in zip(prog.parameters, logits.squeeze().tolist())})
        prog_result = self.fraud_engine.run(prog)
        prob = prog_result.state.expectation_value("I0I1")
        return torch.tensor(prob, device=features.device).unsqueeze(-1)


__all__ = ["HybridQLSTM"]
