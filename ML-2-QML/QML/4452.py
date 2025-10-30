import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import qiskit
import torchquantum as tq

class HybridModel(nn.Module):
    """
    Hybrid quantum‑classical model that exposes the same API as the classical
    counterpart but replaces dense heads with parameterised quantum circuits.
    """

    def __init__(self, mode: str = "regression", n_features: int = 1,
                 n_qubits: int = 2, n_wires: int = 4, device: str = "cpu"):
        super().__init__()
        self.mode = mode
        self.device = device
        self.n_wires = n_wires

        if mode == "fcl":
            # One‑qubit parameterised circuit
            self.circuit = qiskit.QuantumCircuit(1)
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(0)
            self.circuit.ry(self.theta, 0)
            self.circuit.measure_all()
            self.backend = qiskit.Aer.get_backend("qasm_simulator")
            self.shots = 100
        elif mode == "regression":
            self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{n_wires}xRy"])
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(n_wires)))
            self.measure = tq.MeasureAll(tq.PauliZ)
            self.head = nn.Linear(n_wires, 1)
        elif mode == "classification":
            self.cnn = nn.Sequential(
                nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Dropout2d(p=0.5),
                nn.Flatten(),
                nn.Linear(55815, 120),
                nn.ReLU(),
                nn.Dropout2d(p=0.5),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84, 1),
            )
            # Quantum head: 1‑qubit expectation
            self.quantum_head = self._build_fcl_quantum_head()
        elif mode == "quanvolution":
            self.qfilter = self._build_quanvolution_filter(n_wires)
            self.linear = nn.Linear(4 * 14 * 14, 10)
        else:
            raise ValueError(f"Unsupported mode {mode}")

    def _build_fcl_quantum_head(self):
        circ = qiskit.QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circ.h(0)
        circ.ry(theta, 0)
        circ.measure_all()
        return circ

    def _build_quanvolution_filter(self, n_wires: int):
        class QuanvolutionFilter(tq.QuantumModule):
            def __init__(self):
                super().__init__()
                self.n_wires = n_wires
                self.encoder = tq.GeneralEncoder(
                    [
                        {"input_idx": [0], "func": "ry", "wires": [0]},
                        {"input_idx": [1], "func": "ry", "wires": [1]},
                        {"input_idx": [2], "func": "ry", "wires": [2]},
                        {"input_idx": [3], "func": "ry", "wires": [3]},
                    ]
                )
                self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
                self.measure = tq.MeasureAll(tq.PauliZ)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                bsz = x.shape[0]
                device = x.device
                qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
                x = x.view(bsz, 28, 28)
                patches = []
                for r in range(0, 28, 2):
                    for c in range(0, 28, 2):
                        data = torch.stack(
                            [
                                x[:, r, c],
                                x[:, r, c + 1],
                                x[:, r + 1, c],
                                x[:, r + 1, c + 1],
                            ],
                            dim=1,
                        )
                        self.encoder(qdev, data)
                        self.q_layer(qdev)
                        measurement = self.measure(qdev)
                        patches.append(measurement.view(bsz, 4))
                return torch.cat(patches, dim=1)
        return QuanvolutionFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fcl":
            compiled = qiskit.transpile(self.circuit, self.backend)
            qobj = qiskit.compiler.assemble(
                compiled,
                shots=self.shots,
                parameter_binds=[{self.theta: float(x)}],
            )
            job = self.backend.run(qobj)
            result = job.result()
            counts = result.get_counts(self.circuit)
            probs = np.array(list(counts.values())) / self.shots
            states = np.array(list(counts.keys())).astype(float)
            expectation = np.sum(states * probs)
            return torch.tensor([expectation], device=self.device)
        if self.mode == "regression":
            bsz = x.shape[0]
            qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=self.device)
            self.encoder(qdev, x)
            self.random_layer(qdev)
            features = self.measure(qdev)
            out = self.head(features)
            return out.squeeze(-1)
        if self.mode == "classification":
            logits = self.cnn(x)
            # Simple quantum head: expectation of Z after Ry(theta)
            probs = torch.sigmoid(torch.cos(logits))
            return torch.cat((probs, 1 - probs), dim=-1)
        if self.mode == "quanvolution":
            features = self.qfilter(x)
            logits = self.linear(features)
            return F.log_softmax(logits, dim=-1)

    @staticmethod
    def generate_superposition_data(num_wires: int, samples: int):
        omega_0 = np.zeros(2 ** num_wires, dtype=complex)
        omega_0[0] = 1.0
        omega_1 = np.zeros(2 ** num_wires, dtype=complex)
        omega_1[-1] = 1.0

        thetas = 2 * np.pi * np.random.rand(samples)
        phis = 2 * np.pi * np.random.rand(samples)
        states = np.zeros((samples, 2 ** num_wires), dtype=complex)
        for i in range(samples):
            states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
        labels = np.sin(2 * thetas) * np.cos(phis)
        return states, labels

    @staticmethod
    def regression_dataset(samples: int, num_wires: int):
        states, labels = HybridModel.generate_superposition_data(num_wires, samples)
        return torch.utils.data.TensorDataset(
            torch.tensor(states, dtype=torch.cfloat),
            torch.tensor(labels, dtype=torch.float32),
        )
