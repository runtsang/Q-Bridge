"""Importable quantum module defining UnifiedSelfAttention.

This module uses Qiskit to implement a variational self‑attention circuit,
a parameterised fully‑connected layer, an estimator for expectation values,
and a quantum kernel that can be fused into classical pipelines.
"""

import numpy as np
import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from typing import Iterable, Sequence, Callable, Any

# ----- Quantum primitives ---------------------------------------------
class _QuantumAttentionCircuit:
    """Variational circuit that encodes rotation and entanglement parameters."""
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.backend: Backend | None = None
        self._rot_params: np.ndarray | None = None
        self._ent_params: np.ndarray | None = None

    def set_backend(self, backend: Backend) -> None:
        self.backend = backend

    def set_params(self, rot: np.ndarray | None, ent: np.ndarray | None) -> None:
        self._rot_params = rot
        self._ent_params = ent

    def _build(self) -> QuantumCircuit:
        circ = QuantumCircuit(self.n_qubits, self.n_qubits)
        if self._rot_params is None or self._ent_params is None:
            raise ValueError("Rotation or entanglement parameters not set.")
        for i in range(self.n_qubits):
            circ.rx(self._rot_params[3 * i], i)
            circ.ry(self._rot_params[3 * i + 1], i)
            circ.rz(self._rot_params[3 * i + 2], i)
        for i in range(self.n_qubits - 1):
            circ.cx(i, i + 1)
        circ.measure_all()
        return circ

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            shots: int = 1024,
            backend: Backend | None = None) -> np.ndarray:
        self.set_params(rotation_params, entangle_params)
        if backend is None:
            if self.backend is None:
                self.backend = Aer.get_backend("qasm_simulator")
            backend = self.backend
        circ = self._build()
        job = backend.run(circ, shots=shots)
        result = job.result().get_counts(circ)
        bits = np.array([int(k, 2) for k in result.keys()], dtype=float)
        return np.mean(bits / (2 ** self.n_qubits))

class _QuantumFCL:
    """Fully‑connected layer implemented as a single‑qubit parameterised circuit."""
    def __init__(self, n_qubits: int, backend: Backend):
        self.n_qubits = n_qubits
        self.backend = backend
        self.theta = Parameter("theta")
        self.circ = QuantumCircuit(n_qubits)
        self.circ.h(range(n_qubits))
        self.circ.barrier()
        self.circ.ry(self.theta, range(n_qubits))
        self.circ.measure_all()

    def run(self, thetas: Sequence[float]) -> np.ndarray:
        if len(thetas)!= self.n_qubits:
            raise ValueError("Theta count mismatch.")
        param_bindings = [{self.theta: theta} for theta in thetas]
        job = self.backend.run(self.circ, shots=100, parameter_binds=param_bindings)
        result = job.result().get_counts(self.circ)
        counts = np.array(list(result.values()), dtype=float)
        states = np.array(list(result.keys()), dtype=float)
        probabilities = counts / 100
        expectation = np.sum(states * probabilities)
        return np.array([expectation])

class _FastBaseEstimator:
    """Estimator for a parameterised quantum circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._params = list(circuit.parameters)

    def _bind(self, values: Sequence[float]) -> QuantumCircuit:
        if len(values)!= len(self._params):
            raise ValueError("Parameter count mismatch.")
        mapping = dict(zip(self._params, values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(self,
                 observables: Iterable[Any],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[complex]]:
        results: list[list[complex]] = []
        for vals in parameter_sets:
            bound = self._bind(vals)
            state = Statevector.from_instruction(bound)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class _QuantumKernel:
    """Quantum kernel based on a fixed ansatz of Ry rotations."""
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.backend = Aer.get_backend("statevector_simulator")

    def _encode(self, x: np.ndarray) -> QuantumCircuit:
        qc = QuantumCircuit(self.n_qubits)
        for i, val in enumerate(x):
            qc.ry(val, i)
        return qc

    def kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        qc_x = self._encode(x)
        qc_y = self._encode(-y)
        # Swap test circuit
        qc = QuantumCircuit(self.n_qubits + 1)
        qc.h(0)
        qc.append(qc_x, range(1, self.n_qubits + 1))
        qc.append(qc_y, range(1, self.n_qubits + 1))
        qc.cx(0, 1)
        qc.h(0)
        qc.measure(0, 0)
        job = self.backend.run(qc, shots=1024)
        counts = job.result().get_counts(qc)
        prob0 = counts.get("0", 0) / 1024
        return 2 * prob0 - 1

    def matrix(self,
               a: Sequence[np.ndarray],
               b: Sequence[np.ndarray]) -> np.ndarray:
        return np.array([[self.kernel(x, y) for y in b] for x in a])

# ----- Unified module -----------------------------------------------
class UnifiedSelfAttention:
    """Hybrid self‑attention that can run entirely on a quantum backend."""
    def __init__(self,
                 n_qubits: int = 4,
                 use_quantum: bool = False,
                 backend: Backend | None = None):
        self.n_qubits = n_qubits
        self.use_quantum = use_quantum
        self.backend = backend or Aer.get_backend("qasm_simulator")

        # Quantum sub‑modules
        self.quantum_attention = _QuantumAttentionCircuit(n_qubits)
        self.quantum_attention.set_backend(self.backend)
        self.fcl = _QuantumFCL(n_qubits, self.backend)
        self.estimator = _FastBaseEstimator(self.fcl.circuit)
        self.kernel = _QuantumKernel(n_qubits)

    def set_parameters(self,
                       rotation_params: np.ndarray | None = None,
                       entangle_params: np.ndarray | None = None) -> None:
        self.rotation_params = rotation_params
        self.entangle_params = entangle_params

    def run(self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
            shots: int = 1024,
            backend: Backend | None = None) -> np.ndarray:
        self.set_parameters(rotation_params, entangle_params)
        if self.use_quantum:
            attn_value = self.quantum_attention.run(rotation_params,
                                                    entangle_params,
                                                    shots=shots,
                                                    backend=backend)
            fcl_out = self.fcl.run([attn_value[0]])
            return fcl_out
        else:
            return np.array([self.kernel.kernel(inputs, inputs)])

    def fcl(self, x: np.ndarray) -> np.ndarray:
        return self.fcl.run([x])

    def estimate(self,
                 observables: Iterable[Any],
                 parameter_sets: Sequence[Sequence[float]]) -> list[list[complex]]:
        return self.estimator.evaluate(observables, parameter_sets)

    def kernel_matrix(self,
                      a: Sequence[np.ndarray],
                      b: Sequence[np.ndarray]) -> np.ndarray:
        return self.kernel.matrix(a, b)
