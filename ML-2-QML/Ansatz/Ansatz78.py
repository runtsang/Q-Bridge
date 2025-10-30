"""
RealAmplitudesSymmetry ansatz builder with optional parity symmetry.

This module implements a symmetry‑constrained variant of the classic
RealAmplitudes ansatz.  When the ``symmetry`` flag is True (default)
the rotation angles are shared between qubits that are mirror images of
one another about the centre of the register.  For an odd number of
qubits the central qubit receives its own independent angle.  The
entanglement pattern and optional barriers are identical to the
unconstrained ansatz.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec."""
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


def real_amplitudes_symmetry(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a symmetry‑constrained RealAmplitudes circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement layers.  A final rotation layer is added
        unless ``skip_final_rotation_layer`` is True.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of the two‑qubit entangling gates.  Supported
        string values are ``"full"``, ``"linear"``, and ``"circular"``.
    skip_final_rotation_layer : bool, default False
        If True, omit the rotation layer that normally follows the last
        entanglement layer.
    insert_barriers : bool, default False
        If True, insert barriers between layers for clearer visualisation.
    parameter_prefix : str, default "theta"
        Prefix used for the ``ParameterVector`` naming.
    symmetry : bool, default True
        When True, rotation angles are mirrored across the centre of the
        qubit register, halving the number of trainable parameters.
    name : str | None, default None
        Optional circuit name.  If None, defaults to
        ``"RealAmplitudesSymmetry"``.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  The circuit exposes two
        attributes:

        * ``input_params`` – a ``ParameterVector`` containing the
          trainable parameters.
        * ``num_rot_layers`` – the number of rotation layers in the
          circuit (including the optional final layer).

    Notes
    -----
    The symmetry constraint mirrors rotation angles: for qubit index
    ``q`` the parameter index is ``min(q, n-1-q)``.  For even numbers
    of qubits, each pair of mirror qubits shares a single angle; for
    odd numbers, the central qubit has its own independent angle.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")
    if not isinstance(symmetry, bool):
        raise ValueError("symmetry flag must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesSymmetry")

    params_per_layer = n if not symmetry else (n + 1) // 2
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    total_params = params_per_layer * num_rot_layers

    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * params_per_layer
        for q in range(n):
            idx = min(q, n - 1 - q) if symmetry else q
            qc.ry(params[base + idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for i, j in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    return qc


class RealAmplitudesSymmetry(QuantumCircuit):
    """Class‑style wrapper for the symmetry‑constrained RealAmplitudes ansatz.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    reps : int, default 1
        Number of entanglement layers.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Entanglement specification.
    skip_final_rotation_layer : bool, default False
        Omit final rotation layer.
    insert_barriers : bool, default False
        Insert barriers between layers.
    parameter_prefix : str, default "theta"
        Prefix for the parameters.
    symmetry : bool, default True
        Enable the parity‑symmetry constraint.
    name : str, default "RealAmplitudesSymmetry"
        Circuit name.

    Notes
    -----
    The instance behaves like a regular ``QuantumCircuit``.  It
    exposes ``input_params`` and ``num_rot_layers`` attributes inherited
    from the underlying function.
    """
    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = True,
        name: str = "RealAmplitudesSymmetry",
    ) -> None:
        built = real_amplitudes_symmetry(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            symmetry=symmetry,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesSymmetry", "real_amplitudes_symmetry"]
