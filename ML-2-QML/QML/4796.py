import numpy as np
import torch
import torch.nn as nn
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import Sampler

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    use_filter: bool = False,
    filter_kernel_size: int = 2,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """
    Construct a data‑encoded variational circuit that mirrors the
    classical interface.  If `use_filter` is True, a small
    quanvolution‑style circuit is created for each 2×2 patch; otherwise
    a single large circuit encodes the entire input vector.
    """
    if use_filter:
        # Build a reusable 2×2 patch circuit
        patch_circ = QuantumCircuit(filter_kernel_size ** 2)
        params = ParameterVector("x", filter_kernel_size ** 2)
        for i, p in enumerate(params):
            patch_circ.rx(p, i)
        patch_circ.barrier()
        # Random variational layer
        for _ in range(depth):
            for i in range(filter_kernel_size ** 2):
                patch_circ.ry(ParameterVector(f"theta_{_}_{i}", 1)[0], i)
            for i in range(filter_kernel_size ** 2 - 1):
                patch_circ.cz(i, i + 1)
        patch_circ.measure_all()
        return patch_circ, params, None, []

    else:
        # Full‑size circuit
        circuit = QuantumCircuit(num_qubits)
        data_params = ParameterVector("x", num_qubits)
        for i, p in enumerate(data_params):
            circuit.rx(p, i)
        circuit.barrier()
        for _ in range(depth):
            for i in range(num_qubits):
                circuit.ry(ParameterVector(f"theta_{_}_{i}", 1)[0], i)
            for i in range(num_qubits - 1):
                circuit.cz(i, i + 1)
        circuit.measure_all()
        observables = [
            SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
            for i in range(num_qubits)
        ]
        return circuit, data_params, None, observables


class QuantumClassifierModel(nn.Module):
    """
    Quantum‑variational classifier that exposes the same interface as
    its classical counterpart.  The circuit encodes the input vector
    with RX rotations, applies a depth‑controlled ansatz, and extracts
    Pauli‑Z expectation values.  An optional quanvolutional feature
    extractor can be enabled to process 2×2 patches before the
    classical linear head.
    """
    def __init__(
        self,
        num_qubits: int = 784,
        depth: int = 2,
        use_filter: bool = False,
        filter_kernel_size: int = 2,
        shots: int = 1024,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.use_filter = use_filter
        self.shots = shots

        if use_filter:
            # Build a reusable patch circuit
            self.patch_circ, self.patch_params, _, _ = build_classifier_circuit(
                num_qubits=num_qubits,
                depth=depth,
                use_filter=True,
                filter_kernel_size=filter_kernel_size,
            )
            # Classical head maps 14×14 patch features to 2 classes
            self.classical_head = nn.Linear((num_qubits // (filter_kernel_size ** 2)), 2)
        else:
            self.circuit, self.data_params, _, self.observables = build_classifier_circuit(
                num_qubits=num_qubits,
                depth=depth,
                use_filter=False,
            )
            # Classical head maps full‑size quantum features to 2 classes
            self.classical_head = nn.Linear(num_qubits, 2)

        self.backend = Aer.get_backend("qasm_simulator")

    def _run_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """
        Execute the quanvolution patch circuit for a single 2×2 patch.
        Returns the average probability of measuring |1⟩ across the
        four qubits.
        """
        batch = patch.size(0)
        results = torch.zeros(batch, device=patch.device)
        for i in range(batch):
            # Bind data parameters
            param_bind = {
                self.patch_params[j]: patch[i, j].item() for j in range(len(self.patch_params))
            }
            job = execute(
                self.patch_circ,
                self.backend,
                shots=self.shots,
                parameter_binds=[param_bind],
            )
            counts = job.result().get_counts(self.patch_circ)
            # Compute average number of |1⟩ outcomes
            ones = 0
            total = 0
            for bitstring, c in counts.items():
                total += c
                ones += c * sum(int(b) for b in bitstring)
            results[i] = ones / (total * len(self.patch_params))
        return results

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.  `x` is expected to be a 2‑D tensor of shape
        (batch, num_qubits).  For the filter case, `x` is reshaped to
        (batch, 28, 28) and each 2×2 patch is processed with the
        quanvolution circuit.  For the full‑size case, the entire
        vector is encoded into a single circuit.
        """
        batch = x.size(0)
        if self.use_filter:
            # Reshape to image and split into 2×2 patches
            image = x.view(batch, 28, 28)
            patch_features = []
            for i in range(0, 28, 2):
                for j in range(0, 28, 2):
                    patch = image[:, i : i + 2, j : j + 2].reshape(batch, -1)
                    patch_features.append(self._run_patch(patch))
            features = torch.stack(patch_features, dim=1)
        else:
            # Full‑size circuit
            results = torch.zeros(batch, self.num_qubits, device=x.device)
            for i in range(batch):
                param_bind = {self.data_params[j]: x[i, j].item() for j in range(self.num_qubits)}
                job = execute(
                    self.circuit,
                    self.backend,
                    shots=self.shots,
                    parameter_binds=[param_bind],
                )
                counts = job.result().get_counts(self.circuit)
                # Compute Pauli‑Z expectation values
                for q in range(self.num_qubits):
                    ones = 0
                    total = 0
                    for bitstring, c in counts.items():
                        total += c
                        ones += c * int(bitstring[self.num_qubits - q - 1])
                    results[i, q] = (ones - (total - ones)) / total
            features = results

        logits = self.classical_head(features)
        return logits

__all__ = ["QuantumClassifierModel"]
