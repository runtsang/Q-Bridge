"""Quantum estimator that evaluates expectation values of parameterised circuits.

The API mirrors the classical HybridEstimator, enabling seamless switching
between classical and quantum models.  The estimator supports
batched evaluation, optional shot noise, and a helper to build a
photonic fraud‑detection program when strawberryfields is available.
"""

import numpy as np
from typing import Iterable, List, Sequence, Optional
import qiskit
from qiskit import QuantumCircuit as QC, transpile, assemble
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator

ObservableSequence = Iterable[BaseOperator]
ParameterSet = Sequence[float]
ResultSequence = List[List[complex]]

class HybridEstimator:
    """Evaluate a Qiskit QuantumCircuit for batches of parameters.

    The estimator accepts a parameterised circuit that contains a single
    symbolic parameter.  The `evaluate` method returns the expectation
    values of the supplied observables for each parameter set.
    """

    def __init__(self, circuit: QC, backend: Optional[AerSimulator] = None, shots: int = 1024) -> None:
        self.circuit = circuit
        self.backend = backend or AerSimulator()
        self.shots = shots
        self.param = next(iter(circuit.parameters), None)
        if self.param is None:
            raise ValueError("Circuit must contain at least one parameter.")
        # Pre‑compile the circuit for speed
        self._compiled = transpile(self.circuit, self.backend)

    def evaluate(
        self,
        observables: ObservableSequence,
        parameter_sets: Sequence[ParameterSet],
        *,
        shots: int | None = None,
    ) -> ResultSequence:
        """Compute expectation values for each parameter set and observable."""
        if shots is not None:
            self.shots = shots
        observables = list(observables) or [qiskit.quantum_info.PauliZ()]
        results: ResultSequence = []
        for values in parameter_sets:
            # Bind parameters
            binds = [{self.param: v} for v in values]
            qobj = assemble(
                self._compiled,
                shots=self.shots,
                parameter_binds=binds,
            )
            job = self.backend.run(qobj)
            result = job.result()
            # Build statevector for expectation
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def _bind(self, values: Sequence[float]) -> QC:
        """Return a new circuit with parameters bound to the given values."""
        if len(values)!= 1:
            raise ValueError("Only single‑parameter circuits are supported in this simplified estimator.")
        return self.circuit.assign_parameters({self.param: values[0]}, inplace=False)

    # ------------------------------------------------------------------
    # Helper to build a photonic fraud‑detection program
    # ------------------------------------------------------------------
    @staticmethod
    def photonic_fraud_detection_program(
        input_params: "FraudLayerParameters",
        layers: Iterable["FraudLayerParameters"],
        *,
        clip: bool = True,
    ) -> "sf.Program":
        """Return a StrawberryFields program for fraud detection.

        The function imports strawberryfields lazily so that the module can
        be imported even if the dependency is missing.
        """
        try:
            import strawberryfields as sf
            from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
            from dataclasses import dataclass
        except ImportError as exc:
            raise ImportError("strawberryfields is required for photonic fraud detection") from exc

        @dataclass
        class FraudLayerParameters:
            bs_theta: float
            bs_phi: float
            phases: tuple[float, float]
            squeeze_r: tuple[float, float]
            squeeze_phi: tuple[float, float]
            displacement_r: tuple[float, float]
            displacement_phi: tuple[float, float]
            kerr: tuple[float, float]

        def _clip(value: float, bound: float) -> float:
            return max(-bound, min(bound, value))

        def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
            BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
            for i, phase in enumerate(params.phases):
                Rgate(phase) | modes[i]
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
            for i, k in enumerate(params.kerr):
                Kgate(k if not clip else _clip(k, 1)) | modes[i]

        program = sf.Program(2)
        with program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

    # ------------------------------------------------------------------
    # Helper to create a simple quantum hybrid head
    # ------------------------------------------------------------------
    @staticmethod
    def quantum_hybrid_head(
        n_qubits: int,
        shots: int = 1024,
        shift: float = np.pi / 2,
    ) -> "Hybrid":
        """Return a lightweight hybrid head that can be inserted into a classical
        network.  The head performs a Ry rotation and measures the expectation
        value of Pauli‑Z.  It mimics the behaviour of the QML Hybrid class.
        """
        class Hybrid:
            def __init__(self) -> None:
                self.circuit = QC(n_qubits)
                all_qubits = list(range(n_qubits))
                theta = qiskit.circuit.Parameter("theta")
                self.circuit.h(all_qubits)
                self.circuit.barrier()
                self.circuit.ry(theta, all_qubits)
                self.circuit.measure_all()
                self.backend = AerSimulator()
                self.shots = shots
                self.theta = theta
                self._compiled = transpile(self.circuit, self.backend)

            def run(self, thetas: np.ndarray) -> np.ndarray:
                qobj = assemble(
                    self._compiled,
                    shots=self.shots,
                    parameter_binds=[{self.theta: t} for t in thetas],
                )
                job = self.backend.run(qobj)
                result = job.result().get_counts()
                def expectation(counts):
                    total = sum(counts.values())
                    exp = 0.0
                    for bitstring, cnt in counts.items():
                        exp += (1 if bitstring == "0"*self.circuit.num_qubits else -1) * cnt / total
                    return exp
                if isinstance(result, list):
                    return np.array([expectation(r) for r in result])
                return np.array([expectation(result)])

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                import torch
                squeezed = torch.squeeze(inputs) if inputs.shape!= torch.Size([1, 1]) else inputs[0]
                return torch.tensor([self.run(squeezed.tolist())])

        return Hybrid()
