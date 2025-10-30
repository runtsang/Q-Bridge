import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from collections.abc import Iterable, Sequence

class HybridQuanvolution:
    """Quantum hybrid model that processes 2×2 image patches with a
    parameterised circuit and returns a feature vector.  The final
    classification layer is simulated classically for quick prototyping.
    """
    def __init__(self,
                 num_classes: int = 10,
                 shots: int | None = None,
                 seed: int | None = None):
        self.num_classes = num_classes
        self.shots = shots
        self.seed = seed
        self.n_wires = 4
        self.rng = np.random.default_rng(seed)
        self.params = self.rng.random(self.n_wires)
        # classical linear head
        self.linear = self.rng.standard_normal((4 * 14 * 14, num_classes))
        self._build_base_circuit()

    def _build_base_circuit(self):
        qr = QuantumRegister(self.n_wires)
        self.base_circuit = QuantumCircuit(qr)
        # simple entanglement chain
        for i in range(self.n_wires - 1):
            self.base_circuit.cx(i, i + 1)

    def _encode_patch(self, qc: QuantumCircuit, patch: np.ndarray) -> QuantumCircuit:
        for i, val in enumerate(patch):
            qc.ry(val, i)
        return qc

    def _apply_variational(self, qc: QuantumCircuit, params: np.ndarray) -> QuantumCircuit:
        for i, p in enumerate(params):
            qc.rx(p, i)
        return qc

    def _extract_patches(self, img: np.ndarray) -> np.ndarray:
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = img[r:r+2, c:c+2].reshape(-1)
                patches.append(patch)
        return np.array(patches)

    def forward(self, img: np.ndarray) -> np.ndarray:
        """Return a 4×14×14 feature vector obtained from expectation values
        of Pauli‑Z on the first qubit for every patch.
        """
        patches = self._extract_patches(img)
        features = []
        for patch in patches:
            qc = self.base_circuit.copy()
            qc = self._encode_patch(qc, patch)
            qc = self._apply_variational(qc, self.params)
            state = Statevector.from_instruction(qc)
            val = state.expectation_value('Z')
            features.append(val)
        return np.array(features)

    def evaluate(self,
                 observables: Iterable[BaseOperator],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[complex]]:
        """Compute expectation values of the supplied observables for each
        parameter set.  Each set of parameters is applied to the variational
        layer of the circuit that processes every patch; the results are then
        averaged over all patches.
        """
        results: list[list[complex]] = []
        dummy_img = np.zeros((28, 28))
        patches = self._extract_patches(dummy_img)
        for params in parameter_sets:
            params_np = np.array(params)
            row: list[complex] = []
            for obs in observables:
                vals = []
                for patch in patches:
                    qc = self.base_circuit.copy()
                    qc = self._encode_patch(qc, patch)
                    qc = self._apply_variational(qc, params_np)
                    state = Statevector.from_instruction(qc)
                    vals.append(state.expectation_value(obs))
                row.append(sum(vals) / len(vals))
            results.append(row)
        if self.shots is not None:
            results = self._add_noise(results)
        return results

    def _add_noise(self, results: list[list[complex]]) -> list[list[complex]]:
        noisy: list[list[complex]] = []
        for row in results:
            noisy_row = [complex(np.random.normal(float(val.real), max(1e-6, 1 / np.sqrt(self.shots)))) for val in row]
            noisy.append(noisy_row)
        return noisy

__all__ = ["HybridQuanvolution"]
