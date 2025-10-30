from __future__ import annotations

from typing import Iterable, Tuple, List

import pennylane as qml
import pennylane.numpy as np

def build_classifier_circuit(
    num_qubits: int,
    depth: int,
    feature_map: str = "ry"
) -> Tuple[qml.QNode, List[np.ndarray], List[qml.operation.Operator]]:
    """
    Construct a variational quantum classifier with a flexible feature‑map
    and an ansatz that includes RZ rotations and CNOT entanglement.

    The returned tuple follows the same contract as the classical helper:

    * ``circuit`` – a PennyLane QNode ready for training.
    * ``params`` – a flat array of initial variational parameters.
    * ``observables`` – a list of Pauli‑Z operators measured at each qubit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits (features) in the circuit.
    depth : int
        Number of ansatz layers.
    feature_map : {"ry", "amplitude"}, default "ry"
        Encoding strategy for classical data.

    Returns
    -------
    qml.QNode
        The variational quantum circuit.
    List[np.ndarray]
        Initial parameters to feed into the QNode.
    List[qml.operation.Operator]
        Observable list for expectation‑value measurement.
    """
    dev = qml.device("default.qubit", wires=num_qubits)

    # Encoding functions
    if feature_map == "ry":
        def encoder(x: np.ndarray) -> None:
            for i, val in enumerate(x):
                qml.RY(val, wires=i)
    elif feature_map == "amplitude":
        def encoder(x: np.ndarray) -> None:
            qml.StatePrep(x, wires=range(num_qubits))
    else:
        raise ValueError(f"Unsupported feature_map: {feature_map}")

    # Ansatz definition
    def ansatz(params: np.ndarray) -> None:
        idx = 0
        for _ in range(depth):
            for q in range(num_qubits):
                qml.RZ(params[idx], wires=q)
                idx += 1
            for q in range(num_qubits - 1):
                qml.CNOT(wires=[q, q + 1])

    @qml.qnode(dev, interface="autograd")
    def circuit(x: np.ndarray, params: np.ndarray) -> List[float]:
        encoder(x)
        ansatz(params)
        return [qml.expval(qml.PauliZ(q)) for q in range(num_qubits)]

    # Initial parameters
    init_params = np.random.randn(num_qubits * depth)

    observables = [qml.PauliZ(q) for q in range(num_qubits)]

    return circuit, init_params, observables

__all__ = ["build_classifier_circuit"]
