import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from Conv import QuanvCircuit
from QuantumClassifierModel import build_classifier_circuit
import qiskit

class HybridQLSTM(nn.Module):
    """
    Quantum‑enhanced LSTM that replaces recurrent gates with
    small parameterised quantum circuits.  It also applies a
    quantum quanvolution filter (``QuanvCircuit``) to each
    time step and finishes with a variational quantum classifier
    head produced by :func:`build_classifier_circuit`.
    """
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

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int,
                 conv_kernel: int = 2, conv_threshold: float = 0.0,
                 classifier_depth: int = 2) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.n_qubits = n_qubits

        # Quantum quanvolution filter
        self.conv = QuanvCircuit(
            filter_size=conv_kernel,
            backend=qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
            threshold=conv_threshold
        )

        # Quantum‑gate LSTM
        self.forget = self.QLayer(n_qubits)
        self.input = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)

        self.linear_forget = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_input = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_update = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_output = nn.Linear(input_dim + hidden_dim, n_qubits)

        # Quantum classifier head
        self.classifier_circuit, self.encoding_params, _, _ = build_classifier_circuit(
            num_qubits=n_qubits, depth=classifier_depth
        )
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.classifier_linear = nn.Linear(n_qubits, 2)

    def _init_states(self, inputs: torch.Tensor,
                     states: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

    def _run_classifier(self, hidden_vector: torch.Tensor) -> torch.Tensor:
        """
        Run the variational circuit for a single hidden state vector and
        return the expectation values of the Z observable on each qubit.
        """
        param_binds = {param: val for param, val in zip(self.encoding_params, hidden_vector.detach().cpu().numpy())}
        job = qiskit.execute(
            self.classifier_circuit,
            self.backend,
            shots=1024,
            parameter_binds=[param_binds]
        )
        result = job.result()
        counts = result.get_counts(self.classifier_circuit)
        exp_vals = []
        total = sum(counts.values())
        for q in range(self.n_qubits):
            exp = 0.0
            for bitstring, cnt in counts.items():
                bit = int(bitstring[-(q + 1)])  # least‑significant bit corresponds to qubit 0
                z = 1.0 if bit == 1 else -1.0
                exp += z * cnt
            exp /= total
            exp_vals.append(exp)
        return torch.tensor(exp_vals, dtype=torch.float32, device=hidden_vector.device)

    def forward(self, inputs: torch.Tensor,
                states: tuple[torch.Tensor, torch.Tensor] | None = None
                ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            # Apply quantum quanvolution filter to classical data
            if self.input_dim >= 4:
                vec = x.detach().cpu().numpy().reshape(2, 2)
                conv_val = self.conv.run(vec)
                conv_tensor = torch.tensor(conv_val, dtype=x.dtype, device=x.device).unsqueeze(0)
                combined = torch.cat([x, hx, conv_tensor], dim=1)
            else:
                combined = torch.cat([x, hx], dim=1)

            f = torch.sigmoid(self.forget(self.linear_forget(combined)))
            i = torch.sigmoid(self.input(self.linear_input(combined)))
            g = torch.tanh(self.update(self.linear_update(combined)))
            o = torch.sigmoid(self.output(self.linear_output(combined)))

            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # shape (seq_len, batch, hidden_dim)
        logits_list = []
        for t in range(outputs.size(0)):
            for b in range(outputs.size(1)):
                hidden_vec = outputs[t, b]
                exp_vals = self._run_classifier(hidden_vec)
                logits = self.classifier_linear(exp_vals)
                logits_list.append(logits)
        logits_tensor = torch.stack(logits_list).view(outputs.size(0), outputs.size(1), -1)
        return logits_tensor, (hx, cx)

__all__ = ["HybridQLSTM"]
