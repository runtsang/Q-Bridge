import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import qiskit
import numpy as np

class HybridQuanvolutionClassifier(tq.QuantumModule):
    """
    Quantum‑classical hybrid classifier that combines a quantum
    quanvolution filter (using a 4‑wire random circuit) with a
    parameterized quantum fully‑connected layer implemented in Qiskit.
    The module exposes a `forward` method that returns log‑softmax
    probabilities, mirroring the classical counterpart for fair
    benchmarking.
    """
    def __init__(self, n_classes: int = 10, n_wires: int = 4) -> None:
        super().__init__()
        # Quantum filter: 4‑wire encoder + random layer
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

        # Classical linear head to match the output dimensionality
        self.fc = nn.Linear(4 * 14 * 14, n_classes)

        # Quantum fully‑connected layer (parameterized single‑qubit circuit)
        self.qfc_circuit, self.qfc_backend = self._build_qfc()

    def _build_qfc(self):
        """
        Build a simple Qiskit circuit that applies an H gate, an
        Ry rotation with a tunable angle, and measures the qubit.
        """
        qc = qiskit.QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(0)
        qc.ry(theta, 0)
        qc.measure_all()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        return qc, backend

    def _run_qfc(self, theta: float, shots: int = 100) -> np.ndarray:
        """
        Execute the Qiskit circuit with a given rotation angle and
        return the expectation value of the measurement.
        """
        job = qiskit.execute(
            self.qfc_circuit,
            self.qfc_backend,
            shots=shots,
            parameter_binds=[{self.qfc_circuit.parameters[0]: theta}],
        )
        result = job.result().get_counts(self.qfc_circuit)
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        probs = counts / shots
        expectation = np.sum(states * probs)
        return np.array([expectation])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: quantum filter → classical linear head → optional
        quantum fully‑connected layer. The quantum FC is applied to the
        first feature of the batch to illustrate its effect.
        """
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Extract 2×2 patches from the input image
        patches = []
        x_flat = x.view(bsz, 28, 28)
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [
                        x_flat[:, r, c],
                        x_flat[:, r, c + 1],
                        x_flat[:, r + 1, c],
                        x_flat[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        quantum_features = torch.cat(patches, dim=1)

        # Classical linear classification
        logits = self.fc(quantum_features)

        # Optional quantum fully‑connected contribution
        theta = 0.5  # a fixed example angle; in practice this would be a learnable parameter
        qfc_out = torch.from_numpy(self._run_qfc(theta)).to(device)
        logits += qfc_out

        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier"]
