from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from typing import Iterable, Tuple, Dict

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Iterable, Iterable, list[SparsePauliOp]]:
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)
    qc = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        qc.rx(param, qubit)
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            qc.ry(weights[idx], qubit)
            idx += 1
        for qubit in range(num_qubits - 1):
            qc.cz(qubit, qubit + 1)
    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return qc, list(encoding), list(weights), observables

class FraudLayerParameters:
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi, displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

def build_fraud_detection_program(input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for l in layers:
            _apply_layer(q, l, clip=True)
    return prog

def _apply_layer(modes, params: FraudLayerParameters, clip: bool):
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

def _clip(value, bound):
    return max(-bound, min(bound, value))

class SamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        inputs2 = ParameterVector("input", 2)
        weights2 = ParameterVector("weight", 4)
        qc2 = QuantumCircuit(2)
        qc2.ry(inputs2[0], 0)
        qc2.ry(inputs2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[0], 0)
        qc2.ry(weights2[1], 1)
        qc2.cx(0, 1)
        qc2.ry(weights2[2], 0)
        qc2.ry(weights2[3], 1)
        self.circuit = qc2
        self.input_params = inputs2
        self.weight_params = weights2
        self.sampler = StatevectorSampler()
        self.model = QiskitSamplerQNN(circuit=qc2, input_params=inputs2, weight_params=weights2, sampler=self.sampler)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model.forward(inputs)

class QLSTM(nn.Module):
    class QLayer(tq.QuantumModule):
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
                if wire == self.n_wires - 1:
                    tqf.cnot(qdev, wires=[wire, 0])
                else:
                    tqf.cnot(qdev, wires=[wire, wire + 1])
            return self.measure(qdev)

    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int):
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        hx, cx = self._init_states(inputs, states)
        outputs = []
        for x in inputs.unbind(dim=0):
            comb = torch.cat([x, hx], dim=1)
            f = torch.sigmoid(self.forget(self.linear_forget(comb)))
            i = torch.sigmoid(self.input(self.linear_input(comb)))
            g = torch.tanh(self.update(self.linear_update(comb)))
            o = torch.sigmoid(self.output(self.linear_output(comb)))
            cx = f * cx + i * g
            hx = o * torch.tanh(cx)
            outputs.append(hx.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs, (hx, cx)

    def _init_states(self, inputs, states):
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class UnifiedHybridModel(nn.Module):
    def __init__(self, num_qubits: int, depth: int, hidden_dim: int, seq_len: int):
        super().__init__()
        self.classifier_circuit, self.enc, self.params, self.obs = build_classifier_circuit(num_qubits, depth)
        self.fraud_program = build_fraud_detection_program(
            FraudLayerParameters(0.5, 0.1, (0.2, 0.3), (0.1, 0.2), (0.0, 0.0), (0.2, 0.3), (0.0, 0.0), (0.0, 0.0)),
            []
        )
        self.sampler = SamplerQNN()
        self.lstm = QLSTM(hidden_dim, hidden_dim, num_qubits)
        self.parameter_registry: Dict[str, object] = {
            "classifier_circuit": self.classifier_circuit,
            "fraud_program": self.fraud_program,
            "sampler": self.sampler,
            "lstm": self.lstm,
        }

    def forward(self, inputs: dict):
        out: dict = {}
        if "features" in inputs:
            from qiskit import Aer
            backend = Aer.get_backend("statevector_simulator")
            bound_inputs = {param: float(val.item()) for param, val in zip(self.enc, inputs["features"])}
            job = backend.run(self.classifier_circuit.bind_parameters(bound_inputs))
            result = job.result()
            vec = result.get_statevector()
            out["classifier"] = torch.tensor(vec, dtype=torch.float32)
        if "fraud_input" in inputs:
            eng = sf.Engine("gaussian_state")
            eng.run(self.fraud_program, args={"modes": inputs["fraud_input"]})
            out["fraud"] = torch.tensor(eng.run(self.fraud_program).samples[0], dtype=torch.float32)
        if "sampler_input" in inputs:
            out["sampler"] = self.sampler(inputs["sampler_input"])
        if "sequence" in inputs:
            seq_out, _ = self.lstm(inputs["sequence"])
            out["lstm"] = seq_out
        return out

__all__ = [
    "build_classifier_circuit",
    "build_fraud_detection_program",
    "SamplerQNN",
    "QLSTM",
    "UnifiedHybridModel",
]
