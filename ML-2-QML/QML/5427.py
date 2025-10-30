import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
import qiskit
from qiskit import assemble, transpile
import numpy as np
from typing import Iterable, Sequence

class PhotonicFraudLayerParameters:
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

def build_photonic_program(
    input_params: PhotonicFraudLayerParameters,
    layers: Iterable[PhotonicFraudLayerParameters],
) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog

def _apply_layer(modes, params: PhotonicFraudLayerParameters, *, clip: bool):
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

class QLSTMQuantumGate(tq.QuantumModule):
    """Small quantum circuit implementing an LSTM gate."""
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
        for wire in range(self.n_wires - 1):
            tqf.cnot(qdev, wires=[wire, wire + 1])
        tqf.cnot(qdev, wires=[self.n_wires - 1, 0])
        return self.measure(qdev)

class QuantumKernel(tq.QuantumModule):
    """Quantum kernel via a fixed ansatz."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.qdev = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(self.n_wires)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.qdev, x)
        self.ansatz(self.qdev, -y)
        return torch.abs(self.qdev.states.view(-1)[0])

class QuantumCircuitWrapper:
    """Two‑qubit parameterised circuit executed on Aer."""
    def __init__(self, n_qubits: int = 2, shots: int = 1024):
        self.circuit = qiskit.QuantumCircuit(n_qubits)
        qubits = list(range(n_qubits))
        theta = qiskit.circuit.Parameter("theta")
        self.circuit.h(qubits)
        self.circuit.ry(theta, qubits)
        self.circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("aer_simulator")
        self.shots = shots
        self.theta = theta

    def run(self, thetas: np.ndarray) -> np.ndarray:
        compiled = transpile(self.circuit, self.backend)
        qobj = assemble(compiled, shots=self.shots,
                        parameter_binds=[{self.theta: th} for th in thetas])
        job = self.backend.run(qobj)
        result = job.result().get_counts()
        def expectation(counts):
            probs = {k: v / self.shots for k, v in counts.items()}
            exp = sum(int(k, 2) * p for k, p in probs.items())
            return exp
        if isinstance(result, list):
            return np.array([expectation(r) for r in result])
        return np.array([expectation(result)])

class HybridQuantumLayer(nn.Module):
    """Hybrid layer that evaluates a quantum expectation."""
    def __init__(self, n_qubits: int = 2, shots: int = 512, shift: float = np.pi/2):
        super().__init__()
        self.circuit = QuantumCircuitWrapper(n_qubits, shots)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        thetas = inputs.detach().cpu().numpy()
        expectations = self.circuit.run(thetas)
        return torch.tensor(expectations, device=inputs.device).float()

class FraudDetectionHybridQuantum(nn.Module):
    """Full hybrid fraud detection model with photonic layers, QLSTM, quantum kernel
    and a quantum hybrid head."""
    def __init__(self,
                 input_params: PhotonicFraudLayerParameters,
                 layer_params: Sequence[PhotonicFraudLayerParameters],
                 n_qubits_lstm: int = 4,
                 n_qubits_head: int = 2):
        super().__init__()
        self.photo = build_photonic_program(input_params, layer_params)
        self.lstm_gate = QLSTMQuantumGate(n_qubits_lstm)
        self.kernel = QuantumKernel()
        self.hybrid_head = HybridQuantumLayer(n_qubits_head)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (batch, seq_len, 2)
        batch, seq_len, _ = seq.shape
        # encode photonic program (placeholder – real execution would require a full backend)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 5})
        states = []
        for _ in range(seq_len):
            eng.run(self.photo)
            state = eng.backend.state
            vec = torch.tensor(state.to_data(), dtype=torch.float32)
            states.append(vec)
        encoded = torch.stack(states, dim=1)  # (batch, seq_len, dim)
        # quantum LSTM gating
        qlstm_out = self.lstm_gate(encoded)
        # quantum kernel similarity with a learned prototype
        proto = self.register_parameter("proto", nn.Parameter(torch.randn_like(qlstm_out)))
        similarity = self.kernel(qlstm_out, proto)
        # quantum hybrid head
        logits = self.hybrid_head(similarity.unsqueeze(-1))
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["PhotonicFraudLayerParameters",
           "build_photonic_program",
           "QLSTMQuantumGate",
           "QuantumKernel",
           "HybridQuantumLayer",
           "FraudDetectionHybridQuantum"]
