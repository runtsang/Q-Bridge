"""Symmetric Real Amplitudes ansatz (parameter sharing across qubits)."""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]],
) -> list[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of entanglement:
        * ``"full"`` – all-to-all pairs.
        * ``"linear"`` – nearest‑neighbour chain.
        * ``"circular"`` – linear chain with a wrap‑around.
        * a custom sequence of pairs.
        * a callable that returns a sequence given ``num_qubits``.

    Returns
    -------
    list[Tuple[int, int]]
        List of (control, target) qubit indices.

    Raises
    ------
    ValueError
        If an unknown string is supplied or a pair contains out‑of‑range indices.
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
        return [(int(i), int(j)) for i, j in pairs]

    pairs = [(int(i), int(j)) for i, j in entanglement]  # type: ignore[arg-type]
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


def symmetric_real_amplitudes(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    parameter_prefix: str = "theta",
    share_params_per_layer: bool = True,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build a RealAmplitudes‑style ansatz with optional global‑parameter sharing.

    The circuit consists of ``reps`` layers of RY rotations followed by a chosen
    entanglement pattern.  Each rotation layer contains either
    * a single shared parameter applied to all qubits (``share_params_per_layer=True``),
    * or an independent parameter per qubit (``share_params_per_layer=False``).

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default=1
        Number of rotation‑entanglement blocks.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]], default="full"
        Entanglement pattern between qubits.
    skip_final_rotation_layer : bool, default=False
        If ``True``, no rotation layer follows the last entanglement.
    insert_barriers : bool, default=False
        Insert a barrier after each rotation and entanglement block.
    parameter_prefix : str, default="theta"
        Prefix for parameter names.
    share_params_per_layer : bool, default=True
        If ``True`` a single parameter per layer is used; otherwise each qubit
        receives its own parameter.
    name : str | None, default=None
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz circuit.  It exposes ``input_params`` and
        ``num_rot_layers`` attributes for introspection.

    Raises
    ------
    ValueError
        If ``num_qubits`` is less than 1 or if ``reps`` is negative.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 0:
        raise ValueError("reps must be non‑negative.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "SymmetricRealAmplitudes")

    # Determine how many rotation layers are present
    num_rot_layers = reps if skip_final_rotation_layer else reps + 1

    # Build the parameter vector
    if share_params_per_layer:
        params = ParameterVector(parameter_prefix, num_rot_layers)
    else:
        params = ParameterVector(parameter_prefix, num_rot_layers * n)

    def _rotation_layer(layer_idx: int) -> None:
        """Apply a rotation layer with optional parameter sharing."""
        if share_params_per_layer:
            angle = params[layer_idx]
            for q in range(n):
                qc.ry(angle, q)
        else:
            base = layer_idx * n
            for q in range(n):
                qc.ry(params[base + q], q)

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


class SymmetricRealAmplitudes(QuantumCircuit):
    """Convenient class wrapper that behaves like Qiskit's ``QuantumCircuit``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        parameter_prefix: str = "theta",
        share_params_per_layer: bool = True,
        name: str = "SymmetricRealAmplitudes",
    ) -> None:
        built = symmetric_real_amplitudes(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            parameter_prefix=parameter_prefix,
            share_params_per_layer=share_params_per_layer,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]


__all__ = ["SymmetricRealAmplitudes", "symmetric_real_amplitudes"]
