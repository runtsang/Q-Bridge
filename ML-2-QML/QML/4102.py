from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Callable, List, Any

import strawberryfields as sf
from strawberryfields import ops
import numpy as np
import qiskit
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        ops.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    ops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        ops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        ops.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        ops.Kgate(k if not clip else _clip(k, 1)) | modes[i]


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Create a Strawberry Fields program for the hybrid fraud detection model."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


class FraudDetectionHybrid:
    """Quantum implementation of the fraud‑detection circuit."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Sequence[FraudLayerParameters],
    ) -> None:
        self.program = build_fraud_detection_program(input_params, layer_params)

    def _bind_params(self, params: Sequence[float]) -> sf.Program:
        """Bind a flat list of parameters to the program."""
        prog = self.program.copy()
        # Assuming each gate has a 'theta' attribute that can be set.
        for gate, val in zip(prog.circuit, params):
            if hasattr(gate, "theta") and gate.theta is not None:
                gate.theta = val
            if hasattr(gate, "phi") and gate.phi is not None:
                gate.phi = val
        return prog

    def run(self, params: Sequence[float]) -> float:
        """Return the expectation of photon number in mode 0."""
        prog = self._bind_params(params)
        sim = sf.backends.FockSimulator(cutoff_dim=5)
        result = sim.run(prog)
        return float(result.state.expectation_value(ops.N(0)))

    def evaluate(
        self,
        observables: Iterable[Callable[[Sequence[int]], float]] | None = None,
        parameter_sets: Sequence[Sequence[float]] = (),
        *,
        shots: int | None = None,
        seed: int | None = None,
    ) -> List[List[float]]:
        """Compute observables for each parameter set, optionally using shots."""
        observables = list(observables) if observables is not None else [lambda sample: sum(sample)]
        rng = np.random.default_rng(seed) if shots is not None else None
        results: List[List[float]] = []
        for params in parameter_sets:
            row: List[float] = []
            if shots is None:
                prog = self._bind_params(params)
                sim = sf.backends.FockSimulator(cutoff_dim=5)
                state = sim.run(prog).state
                for obs in observables:
                    try:
                        val = obs(state)
                    except TypeError:
                        # Fallback: assume observable expects a state vector
                        val = obs(state)
                    row.append(float(val))
            else:
                prog = self._bind_params(params)
                sampler = sf.backends.FockSampler(cutoff_dim=5)
                result = sampler.run(prog, shots=shots)
                samples = result.samples
                for obs in observables:
                    vals = [obs(tuple(sample)) for sample in samples]
                    mean = float(np.mean(vals))
                    if shots is not None:
                        mean = float(rng.normal(mean, max(1e-6, 1 / shots)))
                    row.append(mean)
            results.append(row)
        return results


class FCL:
    """A simple parameterised quantum circuit for a fully‑connected layer."""
    class QuantumCircuit:
        """Simple parameterised quantum circuit for demonstration."""
        def __init__(self, n_qubits, backend, shots):
            self._circuit = qiskit.QuantumCircuit(n_qubits)
            self.theta = qiskit.circuit.Parameter("theta")
            self._circuit.h(range(n_qubits))
            self._circuit.barrier()
            self._circuit.ry(self.theta, range(n_qubits))
            self._circuit.measure_all()
            self.backend = backend
            self.shots = shots

        def run(self, thetas):
            job = qiskit.execute(
                self._circuit,
                self.backend,
                shots=self.shots,
                parameter_binds=[{self.theta: theta} for theta in thetas],
            )
            result = job.result().get_counts(self._circuit)
            counts = np.array(list(result.values()))
            states = np.array(list(result.keys())).astype(float)
            probabilities = counts / self.shots
            expectation = np.sum(states * probabilities)
            return np.array([expectation])

    def __init__(self):
        simulator = qiskit.Aer.get_backend("qasm_simulator")
        self.circuit = self.QuantumCircuit(1, simulator, 100)

    def run(self, thetas):
        return self.circuit.run(thetas)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
    "FCL",
]
