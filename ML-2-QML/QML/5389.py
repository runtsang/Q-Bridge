from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import List, Union

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.primitives import Estimator as QiskitEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

class FastHybridEstimator:
    """Hybrid estimator that evaluates either Qiskit or Strawberry Fields circuits.
    It can construct fraud‑detection photonic programs, Qiskit EstimatorQNNs, or
    simple fully‑connected parameterised circuits."""

    def __init__(
        self,
        circuit_or_program: Union[QuantumCircuit, sf.Program] | None = None,
        *,
        model_type: str | None = None,
        input_params: 'FraudLayerParameters' | None = None,
        layers: Iterable['FraudLayerParameters'] | None = None,
        shots: int | None = None,
        backend: qiskit.providers.Backend | None = None,
    ) -> None:
        if circuit_or_program is not None:
            self._circuit = circuit_or_program
        elif model_type is not None:
            if model_type == "fraud_photonic":
                if input_params is None or layers is None:
                    raise ValueError("Photonic fraud model requires input_params and layers")
                self._circuit = build_fraud_detection_program(input_params, layers)
            elif model_type == "estimatorqnn":
                self._circuit = _build_estimatorqnn()
            elif model_type == "fcl":
                self._circuit = _build_fcl_circuit()
            else:
                raise ValueError(f"Unsupported model_type {model_type!r}")
        else:
            raise ValueError("Either circuit_or_program or model_type must be provided")

        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self._parameter_names = list(self._circuit.parameters) if hasattr(self._circuit, "parameters") else []

    def _bind(self, parameter_values: Sequence[float]) -> Union[QuantumCircuit, sf.Program]:
        if len(parameter_values)!= len(self._parameter_names):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameter_names, parameter_values))
        if isinstance(self._circuit, QuantumCircuit):
            return self._circuit.assign_parameters(mapping, inplace=False)
        else:
            prog = sf.Program(self._circuit.num_modes)
            with prog.context as q:
                for op, params in zip(self._circuit, mapping.values()):
                    op.apply(q, **{})
            return prog

    def evaluate(
        self,
        observables: Iterable[Union[BaseOperator, sf.ops.Operator]],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        observables = list(observables)
        results: List[List[complex]] = []

        for values in parameter_sets:
            bound = self._bind(values)
            if isinstance(bound, QuantumCircuit):
                state = Statevector.from_instruction(bound)
                row = [state.expectation_value(obs) for obs in observables]
            else:
                eng = sf.Engine("gaussian", backend_options={"cutoff_dim": 10})
                eng.run(bound)
                state = eng.backend.statevector
                row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def sample(
        self,
        parameter_sets: Sequence[Sequence[float]],
        *,
        shots: int | None = None,
    ) -> List[dict]:
        """Return raw measurement counts for each parameter set, using the Qiskit backend."""
        if shots is None:
            shots = self.shots or 1024
        results: List[dict] = []
        for values in parameter_sets:
            bound = self._bind(values)
            if not isinstance(bound, QuantumCircuit):
                raise RuntimeError("Sampling is only supported for Qiskit circuits.")
            job = qiskit.execute(
                bound,
                self.backend,
                shots=shots,
            )
            counts = job.result().get_counts(bound)
            results.append(counts)
        return results


__all__ = ["FastHybridEstimator"]

# --- Helper definitions for building specific models ---------------------------------

from dataclasses import dataclass

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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

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

def _build_estimatorqnn() -> QuantumCircuit:
    params1 = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params1[0], 0)
    qc.rx(params1[1], 0)
    return qc

def _build_fcl_circuit() -> QuantumCircuit:
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(theta, 0)
    qc.measure_all()
    return qc
