"""Core circuit factory for a hybrid variational classifier with caching and parameter‑shift gradient."""

from __future__ import annotations

from typing import Iterable, Tuple, List, Dict

import torch
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit.opflow import StateFn, PauliExpectation, CircuitStateFn
from qiskit.opflow.gradients import ParameterShift
from qiskit.opflow import Gradient

# Simple in‑memory circuit cache
_circuit_cache: Dict[Tuple[int, int], QuantumCircuit] = {}

def _create_ansatz(num_qubits: int, depth: int) -> QuantumCircuit:
    """Internal helper to build a layered ansatz with Ry rotations and CZ entanglement."""
    circuit = QuantumCircuit(num_qubits)
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    for q, param in enumerate(encoding):
        circuit.rx(param, q)

    idx = 0
    for _ in range(depth):
        for q in range(num_qubits):
            circuit.ry(weights[idx], q)
            idx += 1
        for q in range(num_qubits - 1):
            circuit.cz(q, q + 1)

    return circuit

def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, List, List, List[SparsePauliOp]]:
    """
    Construct a variational classifier circuit with caching.
    Returns:
        circuit: QuantumCircuit ready for simulation or execution
        encoding: list of Parameter objects for data encoding
        weights: list of Parameter objects for variational parameters
        observables: list of SparsePauliOp representing measurement on each qubit
    """
    key = (num_qubits, depth)
    if key in _circuit_cache:
        circuit = _circuit_cache[key]
    else:
        circuit = _create_ansatz(num_qubits, depth)
        _circuit_cache[key] = circuit

    encoding = list(circuit.parameters)[:num_qubits]
    weights = list(circuit.parameters)[num_qubits:]

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, encoding, weights, observables

def parameter_shift_gradient(circuit: QuantumCircuit, observable: SparsePauliOp) -> torch.Tensor:
    """
    Compute the gradient of the expectation value w.r.t. all variational parameters
    using the parameter‑shift rule. Returns a torch tensor containing the gradients.
    """
    sim = AerSimulator(method="statevector")
    expectation = PauliExpectation()
    circuit_state = CircuitStateFn(circuit)
    obs_state = StateFn(observable, backend=sim) @ circuit_state
    exp_val = expectation.convert(obs_state)

    grad = Gradient(ParameterShift())
    grad_op = grad.convert(exp_val, params=list(circuit.parameters))
    bound_dict = {p: 0.0 for p in circuit.parameters}
    grad_val = grad_op.eval(bound_dict, backend=sim).data
    return torch.tensor(grad_val, dtype=torch.float32)

__all__ = ["build_classifier_circuit", "parameter_shift_gradient"]
