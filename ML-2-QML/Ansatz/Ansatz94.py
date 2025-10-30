"""SymmetricRealAmplitudes ansatz with optional symmetry, parameter sharing, and layer reordering."""
from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """Return a list of two‑qubit pairs according to a simple entanglement spec.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        Entanglement specification; one of ``'full'``, ``'linear'``, ``'circular'`` or a
        custom sequence/callable that yields a list of (i, j) tuples.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of two‑qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of‑range indices.
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
    else:
        pairs = list(entanglement)  # type: ignore[list-item]

    validated: List[Tuple[int, int]] = []
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
        validated.append((int(i), int(j)))
    return validated


def symmetric_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    symmetry: bool = False,
    share_params_across_layers: bool = False,
    layer_reorder: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """Build a SymmetricRealAmplitudes-style ``QuantumCircuit``.

    The ansatz is a controlled variant of the classic RealAmplitudes template.  It
    retains the RY rotation + CX entanglement structure but offers three optional
    modifications:

    * ``symmetry`` – enforce reflection symmetry, i.e. qubits ``i`` and ``n-1-i``
      share the same rotation parameter.
    * ``share_params_across_layers`` – use a single set of rotation parameters for
      every repetition, reducing the total parameter count.
    * ``layer_reorder`` – swap the order of rotation and entanglement within each
      repetition, applying CX gates before the RY rotations.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    reps
        Number of repetition blocks (rotation + entanglement).
    entanglement
        Entanglement topology as described in ``_resolve_entanglement``.
    skip_final_rotation_layer
        If ``True``, omit the final rotation layer that normally follows the last
        entanglement block.
    insert_barriers
        If ``True``, insert a barrier after each rotation or entanglement block.
    parameter_prefix
        Prefix for the automatically generated rotation parameters.
    symmetry
        Enable reflection symmetry across the qubit index axis.
    share_params_across_layers
        Reuse the same rotation parameters for every repetition.
    layer_reorder
        Apply entanglement before rotation within each repetition.
    name
        Optional name for the resulting circuit.

    Returns
    -------
    QuantumCircuit
        A parameterised RealAmplitudes variant with the requested modifications.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if the entanglement specification is
        invalid.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if not isinstance(symmetry, bool):
        raise ValueError("symmetry must be a bool.")
    if not isinstance(share_params_across_layers, bool):
        raise ValueError("share_params_across_layers must be a bool.")
    if not isinstance(layer_reorder, bool):
        raise ValueError("layer_reorder must be a bool.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "SymmetricRealAmplitudes")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    unique_params_per_layer = n if not symmetry else (n + 1) // 2

    total_params = (
        unique_params_per_layer if share_params_across_layers else unique_params_per_layer * num_rot_layers
    )
    params = ParameterVector(parameter_prefix, total_params)

    def _rotation_layer(layer_idx: int) -> None:
        base = 0 if share_params_across_layers else layer_idx * unique_params_per_layer
        for q in range(n):
            param_idx = base + (q if not symmetry else min(q, n - 1 - q))
            qc.ry(params[param_idx], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        if layer_reorder:
            if insert_barriers:
                qc.barrier()
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()
            _rotation_layer(r)
        else:
            _rotation_layer(r)
            if insert_barriers:
                qc.barrier()
            for (i, j) in pairs:
                qc.cx(i, j)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)

    qc.input_params = params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.symmetry = symmetry  # type: ignore[attr-defined]
    qc.share_params_across_layers = share_params_across_layers  # type: ignore[attr-defined]
    qc.layer_reorder = layer_reorder  # type: ignore[attr-defined]
    return qc


class SymmetricRealAmplitudes(QuantumCircuit):
    """Class-style wrapper that behaves like Qiskit's ``SymmetricRealAmplitudes``.

    The wrapper simply constructs the underlying circuit using
    :func:`symmetric_real_amplitudes` and then composes it into ``self``.  All
    public attributes of the built circuit are exposed (``input_params``,
    ``num_rot_layers``, ``symmetry``, ``share_params_across_layers``, ``layer_reorder``).
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        symmetry: bool = False,
        share_params_across_layers: bool = False,
        layer_reorder: bool = False,
        name: str = "SymmetricRealAmplitudes",
    ) -> None:
        built = symmetric_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            symmetry=symmetry,
            share_params_across_layers=share_params_across_layers,
            layer_reorder=layer_reorder,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.symmetry = built.symmetry  # type: ignore[attr-defined]
        self.share_params_across_layers = built.share_params_across_layers  # type: ignore[attr-defined]
        self.layer_reorder = built.layer_reorder  # type: ignore[attr-defined]


__all__ = ["SymmetricRealAmplitudes", "symmetric_real_amplitudes"]
