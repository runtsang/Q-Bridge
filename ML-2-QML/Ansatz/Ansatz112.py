"""
RealAmplitudesCZExtended – an expressive extension of the RealAmplitudesCZ ansatz.

Features
--------
* **Entanglement depth** – repeat the entanglement pattern up to `entanglement_depth`.
* **Alternative entanglement gate** – choose between CZ and iSWAP.
* **Controlled rotations** – optional CX‑controlled Ry gates after each entangling pair.
* **Barrier insertion** – optional barriers between logical layers.
* **Full parameterisation** – the builder exposes all parameters for optimisation.
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
    depth: int,
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  Accepted strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  Callables are
        expected to accept ``num_qubits`` and return an iterable of
        ``(control, target)`` pairs.
    depth : int
        How many times the base entanglement pattern is repeated.
        ``depth=0`` yields no entanglement.

    Returns
    -------
    List[Tuple[int, int]]
        A flat list containing the entanglement pairs, repeated ``depth``
        times.

    Raises
    ------
    ValueError
        If *depth* is negative or if the specification contains
        invalid qubit indices.
    """
    if depth < 0:
        raise ValueError("entanglement_depth must be >= 0.")
    if depth == 0:
        return []

    # Resolve the base pairs
    if isinstance(entanglement, str):
        if entanglement == "full":
            base_pairs = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
        elif entanglement == "linear":
            base_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        elif entanglement == "circular":
            base_pairs = [(i, i + 1) for i in range(num_qubits - 1)]
            if num_qubits > 2:
                base_pairs.append((num_qubits - 1, 0))
        else:
            raise ValueError(f"Unknown entanglement string: {entanglement!r}")
    elif callable(entanglement):
        base_pairs = list(entanglement(num_qubits))
    else:
        base_pairs = [(int(i), int(j)) for (i, j) in entanglement]  # type: ignore[arg-type]
        for (i, j) in base_pairs:
            if i == j:
                raise ValueError("Entanglement pairs must connect distinct qubits.")
            if not (0 <= i < num_qubits and 0 <= j < num_qubits):
                raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")

    # Repeat according to depth
    return base_pairs * depth


def real_amplitudes_cz_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    entanglement_depth: int = 1,
    entanglement_gate: str = "CZ",
    controlled_rotation: bool = False,
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Construct an extended RealAmplitudesCZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int
        Number of repetition layers of the basic rotation + entanglement
        block.  The total number of rotation layers is ``reps`` or
        ``reps + 1`` depending on ``skip_final_rotation_layer``.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the entanglement pattern.  See
        :func:`_resolve_entanglement` for details.
    entanglement_depth : int
        How many times the entanglement pattern is applied per repetition.
    entanglement_gate : str
        Gate used for entanglement.  Accepted values are ``"CZ"`` and
        ``"iSWAP"``.
    controlled_rotation : bool
        If ``True`` a CX‑controlled Ry gate is applied after each
        entanglement pair.  This adds one additional parameter per pair.
    skip_final_rotation_layer : bool
        When ``True`` the final rotation layer after the last repetition
        is omitted.
    insert_barriers : bool
        If ``True`` a barrier is inserted between logical layers.
    parameter_prefix : str
        Prefix for the :class:`ParameterVector` used to name the circuit
        parameters.
    name : str | None
        Optional name for the circuit.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If any input argument is invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be >= 0.")
    if entanglement_gate not in {"CZ", "iSWAP"}:
        raise ValueError('entanglement_gate must be either "CZ" or "iSWAP".')
    if entanglement_depth < 0:
        raise ValueError("entanglement_depth must be >= 0.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesCZExtended")

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(n, entanglement, entanglement_depth)

    # Determine the number of rotation layers
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    base_params = num_rot_layers * n
    ctrl_params = len(pairs) if controlled_rotation else 0
    total_params = base_params + ctrl_params

    params = ParameterVector(parameter_prefix, total_params)

    def _rot(layer: int) -> None:
        """Apply a layer of single‑qubit Ry rotations."""
        base = layer * n
        for q in range(n):
            qc.ry(params[base + q], q)

    def _controlled_rot(control: int, target: int, idx: int) -> None:
        """Apply a CX‑controlled Ry on the target qubit."""
        qc.cx(control, target)
        qc.ry(params[idx], target)
        qc.cx(control, target)

    # Helper for entanglement gate
    if entanglement_gate == "CZ":
        ent_gate = qc.cz
    else:  # iSWAP
        ent_gate = qc.iswap

    # Build the circuit
    ctrl_idx = base_params  # index for controlled‑rotation parameters
    for r in range(reps):
        _rot(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            ent_gate(i, j)
            if controlled_rotation:
                _controlled_rot(i, j, ctrl_idx)
                ctrl_idx += 1
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rot(reps)

    # Attach metadata for introspection
    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.num_entanglement_layers = reps  # type: ignore[attr-defined]
    qc.entanglement_gate = entanglement_gate  # type: ignore[attr-defined]
    qc.entanglement_depth = entanglement_depth  # type: ignore[attr-defined]
    qc.controlled_rotation = controlled_rotation  # type: ignore[attr-defined]

    return qc


class RealAmplitudesCZExtended(QuantumCircuit):
    """
    Subclass wrapper for the extended RealAmplitudesCZ ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, optional
        Number of repetition layers.  Defaults to 1.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], optional
        Entanglement pattern specification.  Defaults to ``"full"``.
    entanglement_depth : int, optional
        How many times the entanglement pattern is applied per repetition.
        Defaults to 1.
    entanglement_gate : str, optional
        Gate used for entanglement: ``"CZ"`` or ``"iSWAP"``.  Defaults to
        ``"CZ"``.
    controlled_rotation : bool, optional
        Whether to insert CX‑controlled Ry gates after each entanglement
        pair.  Defaults to ``False``.
    skip_final_rotation_layer : bool, optional
        Omit the final rotation layer if ``True``.  Defaults to ``False``.
    insert_barriers : bool, optional
        Insert barriers between logical layers if ``True``.  Defaults to
        ``False``.
    parameter_prefix : str, optional
        Prefix for the parameters.  Defaults to ``"theta"``.
    name : str, optional
        Name of the circuit.  Defaults to ``"RealAmplitudesCZExtended"``.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        entanglement_depth: int = 1,
        entanglement_gate: str = "CZ",
        controlled_rotation: bool = False,
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesCZExtended",
    ) -> None:
        built = real_amplitudes_cz_extended(
            num_qubits,
            reps,
            entanglement,
            entanglement_depth,
            entanglement_gate,
            controlled_rotation,
            skip_final_rotation_layer,
            insert_barriers,
            parameter_prefix,
            name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)
        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_entanglement_layers = built.num_entanglement_layers  # type: ignore[attr-defined]
        self.entanglement_gate = built.entanglement_gate  # type: ignore[attr-defined]
        self.entanglement_depth = built.entanglement_depth  # type: ignore[attr-defined]
        self.controlled_rotation = built.controlled_rotation  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesCZExtended", "real_amplitudes_cz_extended"]
