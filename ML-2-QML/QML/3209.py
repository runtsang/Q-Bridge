import numpy as np
import qiskit
import torch
import torchquantum as tq
import torch.nn.functional as F


class HybridQuantConvLayer:
    """
    Quantum counterpart of HybridQuantConvLayer.

    Features:
    - A parameterised Qiskit circuit that simulates a fully‑connected layer.
    - A TorchQuantum quanvolution filter that processes 2×2 image patches.
    """

    class _FullyConnectedCircuit:
        """Parameterized Qiskit circuit for a fully‑connected layer."""

        def __init__(self, n_qubits: int, backend: qiskit.providers.Backend, shots: int = 100):
            self.n_qubits = n_qubits
            self.backend = backend
            self.shots = shots

            self.circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self.circuit.h(range(n_qubits))
            self.circuit.barrier()
            self.circuit.ry(self.theta, range(n_qubits))
            self.circuit.measure_all()

        def run(self, thetas: np.ndarray) -> np.ndarray:
            """Execute the circuit for each theta and return the expectation value."""
            job = qiskit.execute(
                self.circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self.circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probs = counts / self.shots
            expectation = np.sum(states * probs)
            return np.array([expectation])

    class _QuanvolutionFilter(tq.QuantumModule):
        """TorchQuantum implementation of a 2×2 image patch kernel."""

        def __init__(self):
            super().__init__()
            self.n_wires = 4
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

    def __init__(self, n_qubits: int = 4, backend: qiskit.providers.Backend | None = None, shots: int = 100):
        if backend is None:
            backend = qiskit.Aer.get_backend("qasm_simulator")
        self.fcl_circuit = self._FullyConnectedCircuit(n_qubits, backend, shots)
        self.qfilter = self._QuanvolutionFilter()

    def run_fcl(self, thetas: np.ndarray) -> np.ndarray:
        """
        Run the quantum fully‑connected circuit.

        Parameters
        ----------
        thetas : np.ndarray
            Array of theta parameters for the Ry rotations.

        Returns
        -------
        np.ndarray
            Expectation value of the circuit.
        """
        return self.fcl_circuit.run(thetas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the quantum convolution filter to an image batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Quantum‑encoded features of shape (batch, 4 * 14 * 14).
        """
        return self.qfilter(x)

    def classify(self, x: torch.Tensor, thetas: np.ndarray, n_classes: int = 10) -> torch.Tensor:
        """
        End‑to‑end quantum classifier: apply the quantum filter, then the quantum FC circuit,
        and map to class logits via a classical linear head.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch.
        thetas : np.ndarray
            Parameters for the fully‑connected quantum circuit.
        n_classes : int, optional
            Number of output classes.

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, n_classes).
        """
        # Quantum feature extraction
        qfeat = self.forward(x)  # (batch, 4*14*14)
        # Classical linear head (for demonstration)
        linear = torch.nn.Linear(qfeat.shape[1], n_classes)
        logits = linear(qfeat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridQuantConvLayer"]
