"""RealAmplitudes variant with alternating RY/RX rotation layers and parameter sharing across layers."""
from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

__all__ = [
    "RealAmplitudesAlternatingSharedParams",
    "real_amplitudes_alternating_shared_params",
]


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Resolve entanglement specification to a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification. Accepts the predefined strings ``"full"``, ``"linear"``, or ``"circular"``, a
        sequence of qubit pairs, or a callable that returns a sequence given ``num_qubits``.

    Returns
    -------
    List[Tuple[int, int]]
        List of qubit pairs to entangle with CNOT gates.

    Raises
    ------
    ValueError
        If the specification is unknown or contains invalid pairs.
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


def real_amplitudes_alternating_shared_params(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    share_parameters: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Construct a RealAmplitudes ansatz with alternating RY/RX layers.

    This variant optionally shares the same set of rotation parameters across all
    rotation layers, reducing the parameter count by a factor of ``reps``.
    The alternating pattern (RY on even layers, RX on odd layers) is preserved.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of entanglement layers. If ``skip_final_rotation_layer`` is False,
        an additional rotation layer is appended after the last entanglement block.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement pattern. ``"full"``, ``"linear"``, ``"circular"``, a custom list of pairs,
        or a callable that generates pairs.
    skip_final_rotation_layer : bool, optional
        If True, omit the rotation layer after the final entanglement block.
    insert_barriers : bool, optional
        If True, insert a barrier after each rotation and entanglement block for easier debugging.
    parameter_prefix : str, optional
        Prefix for the rotation parameters.
    share_parameters : bool, optional
        If True, reuse the same set of ``num_qubits`` parameters for every rotation layer,
        effectively imposing a global parameter symmetry.
    name : str | None, optional
        Name of the underlying QuantumCircuit.

    Returns
    -------
    QuantumCircuit
        A quantum circuit object with attributes ``input_params`` and ``num_rot_layers``.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesAlternatingSharedParams")

    # Determine number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Parameter vector
    if share_parameters:
        params = ParameterVector(parameter_prefix, n)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rot(layer: int) -> None:
        """Apply a rotation layer of type RY or RX depending on parity."""
        if share_parameters:
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


class RealAmplitudesAlternatingSharedParams(QuantumCircuit):
    """Convenience subclass that builds the shared‑parameter alternating ansatz.

    The class exposes the same initialization signature as the convenience function.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_parameters: bool = False,
        name: str = "RealAmplitudesAlternatingSharedParams",
    ) -> None:
        built = real_amplitudes_alternating_shared_params(
            num_qubits,
            reps,
            entanglement,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            share_parameters,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
