import numpy as np
import qiskit
import torch
import torch.nn as nn
import torchquantum as tq

def generate_2d_data(kernel_size: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Same synthetic 2‑D dataset used in the classical model.
    """
    data = np.random.uniform(-1.0, 1.0, size=(samples, kernel_size, kernel_size)).astype(np.float32)
    angles = data.reshape(samples, -1).sum(axis=1)
    labels = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return data, labels.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset that returns a 2‑D patch and its regression target."""
    def __init__(self, samples: int, kernel_size: int):
        self.states, self.labels = generate_2d_data(kernel_size, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuanvCircuit:
    """
    Quantum convolution filter implemented with Qiskit.
    It applies a random circuit to each qubit and measures the probability
    of observing |1>.  The output is a single scalar that is later expanded
    into a feature vector for the variational circuit.
    """
    def __init__(self, kernel_size: int, backend, shots: int, threshold: float):
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2‑D patch.
        Parameters
        ----------
        data : np.ndarray
            Shape ``(kernel_size, kernel_size)``.
        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {self.theta[i]: (np.pi if val > self.threshold else 0) for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = qiskit.execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

class QModel(tq.QuantumModule):
    """
    Quantum regression model that first applies a Qiskit quanvolution filter,
    then maps the resulting scalar into a feature vector, and finally
    processes it with a variational circuit implemented in TorchQuantum.
    """
    class QLayer(tq.QuantumModule):
        def __init__(self, num_wires: int):
            super().__init__()
            self.n_wires = num_wires
            self.random_layer = tq.RandomLayer(n_ops=30, wires=list(range(num_wires)))

        def forward(self, qdev: tq.QuantumDevice):
            self.random_layer(qdev)

    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.kernel_size = kernel_size
        self.n_wires = kernel_size ** 2

        # Classical quanvolution filter implemented with Qiskit
        self.encoder = QuanvCircuit(
            kernel_size=kernel_size,
            backend=qiskit.Aer.get_backend("qasm_simulator"),
            shots=100,
            threshold=0.0,
        )

        # Variational circuit
        self.q_layer = self.QLayer(self.n_wires)

        # Measurement and classical head
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(self.n_wires, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Batch of 2‑D patches of shape ``(batch, kernel_size, kernel_size)``.
        Returns
        -------
        torch.Tensor
            Predicted scalar target of shape ``(batch,)``.
        """
        bsz = states.shape[0]

        # Run the classical quanvolution filter on each sample
        features = torch.tensor(
            [self.encoder.run(state.numpy()) for state in states],
            dtype=torch.float32,
            device=states.device,
        )
        # Expand to a full feature vector for the variational circuit
        features = features.unsqueeze(-1).repeat(1, self.n_wires)

        # Prepare quantum device
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=states.device)

        # Encode the classical feature vector into qubit rotations
        for i in range(self.n_wires):
            tq.RY(qdev, params=features[:, i], wires=i)

        # Apply variational layer
        self.q_layer(qdev)

        # Measure and produce output
        out = self.measure(qdev)
        return self.head(out).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_2d_data"]
