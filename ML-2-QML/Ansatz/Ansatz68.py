"""Extended RealAmplitudes ansatz (RY+CX layers with optional RZ mixing)."""
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


def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
    skip_final_rotation_layer: bool = False,
    insert_barriers: bool = False,
    include_rz: bool = False,
    parameter_prefix: str = "theta",
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes circuit with optional RZ mixing layers.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the ansatz.
    reps : int, default=1
        Number of rotation+entanglement repetitions.
    entanglement : str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]]
        Specification of qubit pairs to entangle.  Common strings are
        ``"full"``, ``"linear"``, and ``"circular"``.  Custom sequences or
        callables are also accepted.
    skip_final_rotation_layer : bool, default=False
        If ``True``, omit the final rotation layer after the last
        entanglement step.
    insert_barriers : bool, default=False
        Insert barriers between logical sub‑layers for debugging or
        simulation clarity.
    include_rz : bool, default=False
        If ``True``, add a parameterized RZ rotation on each qubit
        immediately after each entanglement stage, thereby increasing
        the variational expressivity.
    parameter_prefix : str, default="theta"
        Prefix for all parameter names.  Two separate prefixes are
        generated internally: ``{prefix}_rot`` for RY rotations and
        ``{prefix}_phz`` for the optional RZ mixing layers.
    name : str | None, default=None
        Name for the resulting circuit.  Falls back to
        ``"RealAmplitudesExtended"``.

    Returns
    -------
    QuantumCircuit
        Configured ansatz circuit with ``input_params`` and layer
        descriptors attached.

    Notes
    -----
    * The total parameter count is ``(reps + (not skip_final_rotation_layer)) * num_qubits``
      for the RY layers, plus the same number if ``include_rz`` is ``True``.
    * The circuit is fully compatible with Qiskit’s parameter binding
      mechanisms and can be composed or transpiled like any other
      :class:`QuantumCircuit`.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")

    n = int(num_qubits)
    qc = QuantumCircuit(n, name=name or "RealAmplitudesExtended")

    num_rot_layers = reps if skip_final_rotation_layer else reps + 1
    rot_params = ParameterVector(f"{parameter_prefix}_rot", num_rot_layers * n)

    if include_rz:
        phz_params = ParameterVector(f"{parameter_prefix}_phz", num_rot_layers * n)
    else:
        phz_params = []

    all_params = rot_params + phz_params

    def _rotation_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.ry(rot_params[base + q], q)

    def _phase_layer(layer_idx: int) -> None:
        base = layer_idx * n
        for q in range(n):
            qc.rz(phz_params[base + q], q)

    pairs = _resolve_entanglement(n, entanglement)

    for r in range(reps):
        _rotation_layer(r)
        if insert_barriers:
            qc.barrier()
        for (i, j) in pairs:
            qc.cx(i, j)
        if insert_barriers:
            qc.barrier()
        if include_rz:
            _phase_layer(r)
            if insert_barriers:
                qc.barrier()

    if not skip_final_rotation_layer:
        _rotation_layer(reps)
        if include_rz:
            _phase_layer(reps)

    qc.input_params = all_params  # type: ignore[attr-defined]
    qc.num_rot_layers = num_rot_layers  # type: ignore[attr-defined]
    qc.num_phase_layers = num_rot_layers if include_rz else 0  # type: ignore[attr-defined]
    return qc


class RealAmplitudesExtended(QuantumCircuit):
    """Class‑style wrapper that behaves like Qiskit's ``RealAmplitudesExtended``."""

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Sequence[Tuple[int, int]]] = "full",
        skip_final_rotation_layer: bool = False,
        insert_barriers: bool = False,
        include_rz: bool = False,
        parameter_prefix: str = "theta",
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            skip_final_rotation_layer=skip_final_rotation_layer,
            insert_barriers=insert_barriers,
            include_rz=include_rz,
            parameter_prefix=parameter_prefix,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_phase_layers = built.num_phase_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
