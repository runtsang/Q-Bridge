import torch
import torchquantum as tq
import torch.nn.functional as F
from typing import Iterable, Sequence, List, Callable


class QuanvolutionQuantumFilter(tq.QuantumModule):
    """Variational 2×2 patching using a 4‑qubit quantum kernel."""
    def __init__(self, n_wires: int = 4) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=bsz, device=x.device)
        x = x.view(bsz, 28, 28)
        patches: List[torch.Tensor] = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [x[:, r, c], x[:, r, c + 1], x[:, r + 1, c], x[:, r + 1, c + 1]],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                m = self.measure(qdev)
                patches.append(m.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuantumAutoencoder(tq.QuantumModule):
    """Simple variational autoencoder that compresses the 4‑qubit output to 3 qubits."""
    def __init__(self, latent_qubits: int = 3, trash_qubits: int = 2, reps: int = 5) -> None:
        super().__init__()
        self.latent_qubits = latent_qubits
        self.trash_qubits = trash_qubits
        self.ansatz = tq.RandomLayer(
            n_ops=reps * (latent_qubits + trash_qubits),
            wires=list(range(latent_qubits + trash_qubits)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In this toy example we simply return the input as a placeholder.
        # A real implementation would apply the ansatz and measure a compressed state.
        return x


class QuanvolutionHybridQ(tq.QuantumModule):
    """Quantum analogue of the classical hybrid pipeline."""
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = QuanvolutionQuantumFilter()
        self.autoencoder = QuantumAutoencoder()
        # Dummy classifier: a random layer that maps 32‑dimensional quantum output to num_classes
        self.classifier = tq.RandomLayer(n_ops=10, wires=list(range(32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.qfilter(x)
        compressed = self.autoencoder(feats)
        logits = self.classifier(compressed)
        return logits


class FastBaseEstimator:
    """Fast expectation‑value evaluator for a parametric Qiskit circuit."""
    def __init__(self, circuit: tq.QuantumCircuit) -> None:
        self.circuit = circuit
        self.params = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> tq.QuantumCircuit:
        if len(parameter_values)!= len(self.params):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self.params, parameter_values))
        return self.circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[tq.BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> List[List[complex]]:
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = tq.Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results
