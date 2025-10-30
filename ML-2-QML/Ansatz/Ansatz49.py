"""RealAmplitudes variant with alternating RY/RX rotation layers and optional parameter sharing across layers."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Either a predefined string ("full", "linear", "circular") or a custom list/func
        describing the entanglement pairs.

    Returns
    -------
    List[Tuple[int, int]]
        A list of valid qubit pairs for CX gates.

    Raises
    ------
    ValueError
        If an unknown string is provided or an invalid pair is detected.
    """
    if isinstance(entanglement, str):
        if entanglement == "full":
            return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        if entanglement == "linear":
            return [(i, i + 1) for i in range(num_qubits - 1)]
        if entanglement == "circular":
            pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                pairs.append((num_qubits - 1, 0))
            return pairs
        raise ValueError(f"Unknown entanglement string: {entanglement!r}")

    if callable(entanglement):
        pairs = list(entanglement(num_qubits))
        return [(int(i), int(j)) for (i, j) in pairs]

    pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
    for (i, j) in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def real_amplitudes_alternating_controlled(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    share_params_across_layers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """Build a RealAmplitudes variant with alternating RY/RX rotations and optional parameter sharing.

    The circuit alternates between RY and RX rotations on each qubit for each repetition.
    The user can enable *parameter sharing* across all rotation layers to reduce the
    number of variational parameters, which introduces a symmetry that can aid training.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement + rotation blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Defines the entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks for easier visualisation.
    share_params_across_layers : bool, default False
        If True, all rotation layers use the same ParameterVector, reducing the total
        number of parameters from ``n * num_rot_layers`` to ``n``.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str, optional
        Name for the resulting QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        The constructed parameterised circuit with attributes ``input_params`` and
        ``num_rot_layers`` set.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if an invalid entanglement specification is supplied.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingControlled")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    if share_params_across_layers:
        # One shared parameter vector for all layers
        params = ParameterVector(parameter_prefix, n)
    else:
        # Separate parameters for each rotation layer
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply the alternating rotation layer for a given layer index."""
        if share_params_across_layers:
            base = 0
        else:
            base = layer * n
        if layer % 2 == 0:
            for q in range(n):
                qc.ry(params[base + q], q)
        else:
            for q in range(n):
                qc.rx(params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesAlternatingControlled(QuantumCircuit):
    """Class wrapper for the alternating‑rotation variant with optional parameter sharing.

    The class behaves like a standard Qiskit `QuantumCircuit` but exposes two convenience
    attributes:

    * ``input_params`` – the `ParameterVector` used for all rotation layers.
    * ``num_rot_layers`` – the total number of rotation layers (including the optional final one).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement + rotation blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default "full"
        Defines the entanglement pattern.
    skip_final_rotation_layer : bool, default False
        If True, omit the final rotation layer after the last entanglement block.
    insert_barriers : bool, default False
        If True, insert barriers between logical blocks.
    share_params_across_layers : bool, default False
        If True, all rotation layers share the same parameters.
    parameter_prefix : str, default "theta"
        Prefix for the parameter names.
    name : str, default "RealAmplitudesAlternatingControlled"
        Name for the resulting QuantumCircuit.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        share_params_across_layers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesAlternatingControlled",
    ) -> None:
        built = real_amplitudes_alternating_controlled(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            share_params_across_layers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesAlternatingControlled", "real_amplitudes_alternating_controlled"]
