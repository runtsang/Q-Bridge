import torch
import torchquantum as tq
import numpy as np
from torchquantum.circuit import ParameterVector

class HybridQuanvolutionClassifier(tq.QuantumModule):
    """
    Quantum implementation of the hybrid quanvolution classifier.
    Combines:
        * a 2×2 patch encoder with a RandomLayer (QuanvolutionFilter)
        * a single‑qubit quantum FCL (parameterised Ry)
        * a variational classifier circuit (Rx/Ry/CZ layers)
    The forward method returns a log‑softmax over the class logits.
    """
    def __init__(self, n_qubits: int = 4, num_classes: int = 10, depth: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.num_classes = num_classes
        self.depth = depth

        # Quanvolution filter
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_qubits)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)

        # Quantum FCL: single qubit Ry with trainable parameter
        self.fcl_theta = tq.Parameter("theta", shape=(1,))
        self.fcl_circuit = tq.QuantumCircuit(n_qubits)
        self.fcl_circuit.ry(self.fcl_theta, 0)

        # Variational classifier circuit
        self.classifier_circuit = self._build_classifier_circuit(n_qubits, depth)
        self.classifier_obs = [tq.PauliZ(wires=[i]) for i in range(n_qubits)]

    def _build_classifier_circuit(self, n_qubits: int, depth: int):
        encoding = ParameterVector("x", n_qubits)
        weights = ParameterVector("theta", n_qubits * depth)

        qc = tq.QuantumCircuit(n_qubits)
        for param, qubit in zip(encoding, range(n_qubits)):
            qc.rx(param, qubit)

        idx = 0
        for _ in range(depth):
            for qubit in range(n_qubits):
                qc.ry(weights[idx], qubit)
                idx += 1
            for qubit in range(n_qubits - 1):
                qc.cz(qubit, qubit + 1)

        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(self.n_qubits, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, patch)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        features = torch.cat(patches, dim=1)

        # Quantum FCL expectation
        fcl_exp = self.fcl_circuit.expectation(self.fcl_theta, qdev)
        fcl_out = torch.tanh(fcl_exp)

        # Classifier circuit: bind feature parameters
        params = {"x{}".format(i): val for i, val in enumerate(features.squeeze())}
        self.classifier_circuit.bind_parameters(params)

        # Simulate statevector
        backend = tq.Aer.get_backend("statevector_simulator")
        result = backend.run(self.classifier_circuit).result()
        state = result.get_statevector()

        # Compute expectation of PauliZ on each qubit as logits
        logits = torch.tensor([tq.expectation(state, obs) for obs in self.classifier_obs])
        logits = logits + fcl_out  # combine quantum FCL output
        return torch.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionClassifier"]
