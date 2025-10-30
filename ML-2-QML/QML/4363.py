import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import qiskit
from qiskit import execute, Aer
from qiskit.circuit import ParameterVector


class QuantumSampler(nn.Module):
    """Parameterised Qiskit sampler returning 4‑dimensional probability vectors."""
    def __init__(self):
        super().__init__()
        self.inputs = ParameterVector("in", 2)
        self.weights = ParameterVector("w", 4)
        self.circuit = qiskit.QuantumCircuit(2)
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)
        self.circuit.measure_all()
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 1024

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        # inp shape (batch, 2)
        probs = []
        for sample in inp:
            bound = {self.inputs[0]: sample[0], self.inputs[1]: sample[1]}
            job = execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[bound],
            )
            counts = job.result().get_counts(self.circuit)
            probs.append(
                [
                    counts.get("00", 0) / self.shots,
                    counts.get("01", 0) / self.shots,
                    counts.get("10", 0) / self.shots,
                    counts.get("11", 0) / self.shots,
                ]
            )
        return torch.tensor(probs, dtype=torch.float32)


class QuantumQLSTM(nn.Module):
    """Quantum LSTM where each gate is a small quantum circuit."""
    class QLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
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
            qdev = tq.QuantumDevice(
                n_wires=self.n_wires, bsz=x.shape[0], device=x.device
            )
            self.encoder(qdev, x)
            for wire, gate in enumerate(self.params):
                gate(qdev, wires=wire)
            for wire in range(self.n_wires - 1):
                tqf.cnot(qdev, wires=[wire, wire + 1])
            tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits
        self.forget = self.QLayer(n_qubits)
        self.input_gate = self.QLayer(n_qubits)
        self.update = self.QLayer(n_qubits)
        self.output = self.QLayer(n_qubits)
        self.linear_f = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_i = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_u = nn.Linear(input_dim + hidden_dim, n_qubits)
        self.linear_o = nn.Linear(input_dim + hidden_dim, n_qubits)

    def forward(
        self,
        seq: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hx, cx = self._init_states(seq, states)
        outputs = []
        for x in seq.unbind(dim=0):
            combined = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_f(combined)))
            i = torch.sigmoid(self.input_gate(self.linear_i(combined)))
            u = torch.tanh(self.update(self.linear_u(combined)))
            o = torch.sigmoid(self.output(self.linear_o(combined)))
            cx = f * cx + i * u
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        return torch.cat(outputs, dim=0), (hx, cx)

    def _init_states(
        self,
        seq: torch.Tensor,
        states: Tuple[torch.Tensor, torch.Tensor] | None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch = seq.size(1)
        device = seq.device
        return (
            torch.zeros(batch, self.hidden_dim, device=device),
            torch.zeros(batch, self.hidden_dim, device=device),
        )


class QuantumFraudDetection:
    """Photonic fraud‑detection circuit built with StrawberryFields."""
    def __init__(self):
        self.program = None

    def build(self, input_params, layers):
        self.program = sf.Program(2)
        with self.program.context as q:
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)

    def _apply_layer(self, modes, params, *, clip):
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    def _clip(self, value, bound):
        return max(-bound, min(bound, value))

    def run(self, inputs):
        # Placeholder: return mean of inputs as a classical proxy
        return inputs.mean(dim=1, keepdim=True)


class QuantumFCL:
    """Parameterised single‑qubit circuit returning an expectation value."""
    def __init__(self):
        self.backend = Aer.get_backend("qasm_simulator")
        self.shots = 512

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        circuit = qiskit.QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circuit.h(0)
        circuit.ry(theta, 0)
        circuit.measure_all()
        job = execute(
            circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{theta: t} for t in thetas],
        )
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / self.shots
        return np.sum(probs)


class HybridSamplerQLSTMQuantum(nn.Module):
    """Quantum hybrid: sampler → quantum LSTM → fraud detection → FCL."""
    def __init__(self, hidden_dim: int = 8, n_qubits: int = 4):
        super().__init__()
        self.sampler = QuantumSampler()
        self.lstm = QuantumQLSTM(input_dim=4, hidden_dim=hidden_dim, n_qubits=n_qubits)
        self.fraud = QuantumFraudDetection()
        self.fcl = QuantumFCL()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs: Tensor of shape (seq_len, batch, 2)
        """
        probs = self.sampler(inputs.reshape(-1, 2))
        probs = probs.reshape(inputs.shape[0], inputs.shape[1], -1)
        lstm_out, _ = self.lstm(probs)
        fraud_out = self.fraud.run(lstm_out.detach().cpu().numpy())
        return torch.tensor(fraud_out, dtype=torch.float32)


__all__ = ["HybridSamplerQLSTMQuantum"]
