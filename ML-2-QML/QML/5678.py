"""Quantum classifier circuit factory and evaluator.

This module uses Qiskit and the Aer simulator to build a
parameter‑efficient variational circuit and to evaluate
expectation values of a set of Pauli‑Z observables.
"""

from __future__ import annotations

from typing import Iterable, Tuple, List

from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# --------------------------------------------------------------------------- #
# Circuit builder
# --------------------------------------------------------------------------- #
def build_classifier_circuit(
    num_qubits: int,
    depth: int,
) -> Tuple[QuantumCircuit, Iterable, Iterable, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of variational layers.

    Returns
    -------
    circuit : QuantumCircuit
        The parameterised circuit.
    encoding : Iterable[Parameter]
        Parameters that encode the classical data.
    weights : Iterable[Parameter]
        Variational parameters of the ansatz.
    observables : List[SparsePauliOp]
        Pauli‑Z observables used for classification.
    """
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)

    # Data encoding: RX rotations
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    # Variational layers
    idx = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[idx], qubit)
            idx += 1
        # Entangling layer
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    # Observables: Z on each qubit
    observables = [
        SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1))
        for i in range(num_qubits)
    ]

    return circuit, list(encoding), list(weights), observables


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
def evaluate_classifier(
    circuit: QuantumCircuit,
    param_values: torch.Tensor,
    observables: List[SparsePauliOp],
    shots: int = 1024,
) -> torch.Tensor:
    """Execute the circuit for each row of ``param_values`` and return expectation values.

    Parameters
    ----------
    circuit : QuantumCircuit
        The parameterised circuit to run.
    param_values : torch.Tensor
        2‑D tensor of shape (batch, num_params) containing values for
        ``circuit.parameters`` in the same order.
    observables : List[SparsePauliOp]
        Observables to evaluate.
    shots : int, default 1024
        Number of shots for the simulation.

    Returns
    -------
    torch.Tensor
        Expectation values of shape (batch, len(observables)).
    """
    backend = Aer.get_backend("aer_simulator_statevector")
    batch_size = param_values.shape[0]
    results: List[List[float]] = []

    # Convert to numpy for binding
    param_np = param_values.detach().cpu().numpy()

    for i in range(batch_size):
        bound_circuit = circuit.bind_parameters(
            {p: val for p, val in zip(circuit.parameters, param_np[i])}
        )
        trans_circ = transpile(bound_circuit, backend)
        qobj = assemble(trans_circ, backend=backend, shots=shots)
        job = backend.run(qobj)
        statevector = job.result().get_statevector(trans_circ)

        exp_vals = []
        for op in observables:
            exp = (statevector.conj().T @ op.data @ statevector).real
            exp_vals.append(exp)
        results.append(exp_vals)

    return torch.tensor(results, dtype=torch.float32, device=param_values.device)


__all__ = [
    "build_classifier_circuit",
    "evaluate_classifier",
]
