import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap, SparsePauliOp
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
import torchquantum as tq

def QCNN() -> EstimatorQNN:
    algorithm_globals.random_seed = 12345
    estimator = Estimator()
    feature_map = ZFeatureMap(8)
    ansatz = RealAmplitudes(8, reps=1)
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, inplace=True)
    circuit.compose(ansatz, inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=estimator,
    )
    return qnn

def Autoencoder(latent_dim: int = 32) -> SamplerQNN:
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    num_latent = latent_dim
    num_trash = 2
    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(RealAmplitudes(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        auxiliary_qubit = num_latent + 2 * num_trash
        circuit.h(auxiliary_qubit)
        for i in range(num_trash):
            circuit.cswap(auxiliary_qubit, num_latent + i, num_latent + num_trash + i)
        circuit.h(auxiliary_qubit)
        circuit.measure(auxiliary_qubit, cr[0])
        return circuit
    circuit = auto_encoder_circuit(num_latent, num_trash)
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=latent_dim,
        sampler=sampler,
    )
    return qnn

class QuantumQCNN(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.qnn = QCNN()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)

class QuantumAutoencoder(nn.Module):
    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.qnn = Autoencoder(latent_dim=latent_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.qnn(x)

class QuantumFCL(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.backend = Aer.get_backend("qasm_simulator")
        self.num_qubits = 1
        self.qc = QuantumCircuit(self.num_qubits)
        self.theta = qiskit.circuit.Parameter("theta")
        self.qc.h(range(self.num_qubits))
        self.qc.ry(self.theta, range(self.num_qubits))
        self.qc.measure_all()
        self.shots = 100
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        for theta in x.squeeze().tolist():
            job = execute(self.qc, self.backend, shots=self.shots, parameter_binds=[{self.theta: theta}])
            result = job.result()
            counts = result.get_counts(self.qc)
            probs = {state: count / self.shots for state, count in counts.items()}
            exp = sum(float(state) * p for state, p in probs.items())
            results.append(exp)
        return torch.tensor(results, dtype=torch.float32).unsqueeze(-1)

class QuantumQLSTM(nn.Module):
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

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
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

    def _init_states(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None) -> Tuple[torch.Tensor, torch.Tensor]:
        if states is not None:
            return states
        batch_size = inputs.size(1)
        device = inputs.device
        return (
            torch.zeros(batch_size, self.hidden_dim, device=device),
            torch.zeros(batch_size, self.hidden_dim, device=device),
        )

class HybridQLSTM(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        vocab_size: int,
        tagset_size: int,
        latent_dim: int = 32,
    ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.qcnn = QuantumQCNN(embedding_dim)
        self.autoencoder = QuantumAutoencoder(latent_dim=latent_dim)
        self.lstm = QuantumQLSTM(latent_dim, hidden_dim, n_qubits=0)
        self.fcl = QuantumFCL(hidden_dim)

    def forward(self, sentence: torch.Tensor) -> torch.Tensor:
        embeds = self.word_embeddings(sentence)
        batch_size = embeds.size(1)
        seq_len = embeds.size(0)
        x = embeds.view(seq_len * batch_size, -1)
        x = self.qcnn(x)
        latent = self.autoencoder(x)
        latent = latent.view(seq_len, batch_size, -1)
        lstm_out, _ = self.lstm(latent)
        out = self.fcl(lstm_out)
        return out

__all__ = ["QuantumQCNN", "QuantumAutoencoder", "QuantumFCL", "QuantumQLSTM", "HybridQLSTM"]
