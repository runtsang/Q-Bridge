import pennylane as qml
import numpy as np
from typing import Iterable, List, Sequence


class FastBaseEstimatorQML:
    """
    Lightweight estimator for expectation values of a parametrized hybrid circuit.
    Supports optional shot noise via a Gaussian approximation.
    """
    def __init__(self, circuit: qml.QNode):
        self._circuit = circuit

    def _bind(self, params: Sequence[float]) -> np.ndarray:
        # Pennylane QNode accepts a flat parameter array
        return np.array(params, dtype=np.float64)

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[complex]]:
        """
        Compute expectation values for all parameter sets and observables.
        If `shots` is provided, returns noisy estimates using a normal approximation.
        """
        results: List[List[complex]] = []
        rng = np.random.default_rng(seed)
        for params in parameter_sets:
            state = self._circuit(*self._bind(params))
            row = [state.expectation_value(obs) for obs in observables]
            if shots is not None:
                # Gaussian shot‑noise approximation: mean ± 1/√shots
                noisy_row = [rng.normal(v, max(1e-6, 1/np.sqrt(shots))) for v in row]
                row = noisy_row
            results.append(row)
        return results


class QuantumNATHybridQML:
    """
    Hybrid classical‑quantum circuit that mirrors the TorchQuantum implementation
    using Pennylane. The circuit comprises a classical encoder, a random‑layer
    inspired variational block, and measurement of Pauli‑Z observables.
    """
    def __init__(self, n_wires: int = 4, device: str = "default.qubit"):
        self.n_wires = n_wires
        self.dev = qml.device(device, wires=n_wires)
        self._create_qnode()

    def _create_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(*params):
            # Classical encoder: simple linear mapping via rotations
            for i in range(self.n_wires):
                qml.RY(params[i], wires=i)
            # Random entangling layer (fixed for all forward passes)
            for _ in range(10):
                wire = np.random.randint(self.n_wires)
                qml.CNOT(wires=[wire, (wire + 1) % self.n_wires])
            # Trainable rotations
            for i in range(self.n_wires):
                qml.RX(params[self.n_wires + i], wires=i)
            return [qml.expval(qml.PauliZ(w)) for w in range(self.n_wires)]
        self._circuit = circuit

    def evaluate(
        self,
        observables: Iterable[qml.operation.Operator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """
        Execute the hybrid circuit for each parameter set and return expectation values.
        """
        estimator = FastBaseEstimatorQML(self._circuit)
        return estimator.evaluate(observables, parameter_sets)

__all__ = ["QuantumNATHybridQML", "FastBaseEstimatorQML"]
