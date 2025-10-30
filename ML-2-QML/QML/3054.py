import numpy as np
import torch
import torch.nn as nn
import torchquantum as tq
import qiskit
from typing import Iterable

class HybridFCL(tq.QuantumModule):
    """
    Quantum implementation of the hybrid fully connected layer.
    Combines the quantum regression circuit from the second seed
    with a simple parameterized Qiskit circuit for inference.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))
            self.rx = tq.RX(has_params=True, trainable=True)
            self.ry = tq.RY(has_params=True, trainable=True)

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)
            for wire in range(self.n_wires):
                self.rx(qdev, wires=wire)
                self.ry(qdev, wires=wire)

    def __init__(self, num_wires: int = 3):
        super().__init__()
        self.n_wires = num_wires
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict[f"{num_wires}xRy"])
        self.q_layer = self.QLayer(num_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_wires, 1)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        bsz = state_batch.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=state_batch.device)
        self.encoder(qdev, state_batch)
        self.q_layer(qdev)
        features = self.measure(qdev)
        return self.head(features).squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Run a lightweight Qiskit circuit that mirrors the classical
        FCL 'run' interface.  Each theta is bound to a 1‑qubit
        Ry gate, and the expectation of the Z measurement is
        returned.
        """
        circuit = qiskit.QuantumCircuit(1)
        theta = qiskit.circuit.Parameter("theta")
        circuit.h(0)
        circuit.ry(theta, 0)
        circuit.measure_all()
        backend = qiskit.Aer.get_backend("qasm_simulator")
        job = qiskit.execute(
            circuit,
            backend,
            shots=1024,
            parameter_binds=[{theta: t} for t in thetas],
        )
        result = job.result()
        counts = result.get_counts(circuit)
        probs = np.array(list(counts.values())) / 1024
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

__all__ = ["HybridFCL"]
