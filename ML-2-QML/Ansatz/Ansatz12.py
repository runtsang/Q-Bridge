"""Extended RealAmplitudes ansatz builder.

This module implements an enriched version of the classic RealAmplitudes
ansatz.  The new design offers:
  • Choice of rotation gate (RY, RX or RZ).
  • Choice of entanglement gate (CX, CZ or a parametrised RZZ).
  • Optional sharing of rotation or entanglement parameters across layers.
  • Optional barrier insertion between logical layers.
  • Optional skipping of the final rotation or entanglement layer.
  • Flexible entanglement schedules (full, linear, circular, or user‑supplied).

All parameters are validated with clear error messages.  The returned
``QuantumCircuit`` exposes two attributes:
    * ``input_params`` – a flat :class:`~qiskit.circuit.ParameterVector`
      containing *all* parameters in the order they appear in the circuit.
    * ``num_rot_layers`` and ``num_ent_layers`` – the number of rotation
      and entanglement layers, respectively.

The module can be imported and used exactly like the original:

>>> from ansatz_scaled.real_amplitudes_extension import real_amplitudes_extended
>>> qc = real_amplitudes_extended(num_qubits=4, reps=2)
"""

from __future__ import annotations

from typing import Callable, Iterable, List, Sequence, Tuple

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector


# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

def _resolve_entanglement(
    num_qubits: int,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    """
    Resolve an entanglement specification into a list of two‑qubit pairs.

    Parameters
    ----------
    num_qubits
        Number of qubits in the circuit.
    entanglement
        * ``"full"`` – all distinct pairs.
        * ``"linear"`` – nearest‑neighbour pairs.
        * ``"circular"`` – linear + a wrap‑around pair.
        * A custom sequence of integer tuples.
        * A callable that accepts ``num_qubits`` and returns an iterable of
          tuples.

    Returns
    -------
    List[Tuple[int, int]]
        Validated list of qubit pairs.

    Raises
    ------
    ValueError
        If the specification is invalid or contains out‑of-range indices.
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
        pairs = list(entanglement)

    # Ensure pairs are valid
    for i, j in pairs:
        if i == j:
            raise ValueError("Entanglement pairs must connect distinct qubits.")
        if not (0 <= i < num_qubits and 0 <= j < num_qubits):
            raise ValueError(f"Entanglement pair {(i, j)} out of range for n={num_qubits}.")
    return pairs


# --------------------------------------------------------------------------- #
# Main construction routine
# --------------------------------------------------------------------------- #

def real_amplitudes_extended(
    num_qubits: int,
    reps: int = 1,
    entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
    rotation_gate: str = "ry",
    entanglement_gate: str = "cx",
    skip_final_rotation_layer: bool = False,
    skip_final_entanglement_layer: bool = False,
    insert_barriers: bool = False,
    rotation_parameter_prefix: str = "theta",
    entanglement_parameter_prefix: str = "phi",
    share_rotation_parameters: bool = False,
    share_entanglement_parameters: bool = False,
    name: str | None = None,
) -> QuantumCircuit:
    """
    Build an extended RealAmplitudes-style ``QuantumCircuit``.
    Parameters
    ----------
    num_qubits
        Number of qubits.
    reps
        Number of rotation–entanglement cycles.
    entanglement
        Specification of entanglement pairs (see :func:`_resolve_entanglement`).
    rotation_gate
        Gate used for single‑qubit rotations.  One of ``"ry"``, ``"rx"``, ``"rz"``.
    entanglement_gate
        Gate used for two‑qubit entanglement.  One of ``"cx"``, ``"cz"``,
        ``"rzz"``.  ``"rzz"`` requires an additional parameter per pair.
    skip_final_rotation_layer
        If ``True``, the final rotation layer is omitted.
    skip_final_entanglement_layer
        If ``True``, the final entanglement layer is omitted.
    insert_barriers
        Insert a barrier after each logical layer for readability.
    rotation_parameter_prefix
        Prefix for rotation parameters.
    entanglement_parameter_prefix
        Prefix for entanglement parameters (only used with ``"rzz"``).
    share_rotation_parameters
        If ``True``, all rotation layers reuse the same set of parameters.
    share_entanglement_parameters
        If ``True``, all entanglement layers reuse the same set of parameters.
    name
        Optional circuit name.

    Returns
    -------
    QuantumCircuit
        The constructed ansatz.

    Raises
    ------
    ValueError
        If an unsupported gate name is supplied or parameters are inconsistent.
    """
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1.")
    if reps < 1:
        raise ValueError("reps must be >= 1.")

    # Validate gate names
    _rot_gates = {"ry": QuantumCircuit.ry, "rx": QuantumCircuit.rx, "rz": QuantumCircuit.rz}
    if rotation_gate not in _rot_gates:
        raise ValueError(f"Unsupported rotation_gate {rotation_gate!r}. "
                         f"Supported: {list(_rot_gates.keys())}")

    _ent_gates = {"cx": QuantumCircuit.cx, "cz": QuantumCircuit.cz, "rzz": QuantumCircuit.rzz}
    if entanglement_gate not in _ent_gates:
        raise ValueError(f"Unsupported entanglement_gate {entanglement_gate!r}. "
                         f"Supported: {list(_ent_gates.keys())}")

    # Resolve entanglement pairs
    pairs = _resolve_entanglement(num_qubits, entanglement)
    n_pairs = len(pairs)

    # Determine number of rotation and entanglement layers
    rot_layers = reps if skip_final_rotation_layer else reps + 1
    ent_layers = reps if skip_final_entanglement_layer else reps + 1

    # Parameter counts
    num_rot_params = n_pairs if share_rotation_parameters else num_qubits * rot_layers
    num_ent_params = n_pairs if share_entanglement_parameters else n_pairs * ent_layers

    total_params = num_rot_params + num_ent_params
    input_params = ParameterVector("theta", total_params)
    rot_params = input_params[:num_rot_params]
    ent_params = input_params[num_rot_params:]

    # Build circuit
    qc = QuantumCircuit(num_qubits, name=name or "RealAmplitudesExtended")

    # Helper to apply a rotation layer
    def _apply_rotation(layer: int) -> None:
        base = 0 if share_rotation_parameters else layer * num_qubits
        for q in range(num_qubits):
            param = rot_params[base + q]
            _rot_gates[rotation_gate](qc, param, q)

    # Helper to apply an entanglement layer
    def _apply_entanglement(layer: int) -> None:
        base = 0 if share_entanglement_parameters else layer * n_pairs
        for idx, (i, j) in enumerate(pairs):
            if entanglement_gate == "rzz":
                param = ent_params[base + idx]
                _ent_gates[entanglement_gate](qc, param, i, j)
            else:
                _ent_gates[entanglement_gate](qc, i, j)

    # Assemble layers
    for r in range(reps):
        _apply_rotation(r)
        if insert_barriers:
            qc.barrier()
        _apply_entanglement(r)
        if insert_barriers:
            qc.barrier()

    # Final layers
    if not skip_final_rotation_layer:
        _apply_rotation(reps)
    if not skip_final_entanglement_layer:
        _apply_entanglement(reps)

    qc.input_params = input_params  # type: ignore[attr-defined]
    qc.num_rot_layers = rot_layers  # type: ignore[attr-defined]
    qc.num_ent_layers = ent_layers  # type: ignore[attr-defined]
    return qc


# --------------------------------------------------------------------------- #
# Class wrapper
# --------------------------------------------------------------------------- #

class RealAmplitudesExtended(QuantumCircuit):
    """Class wrapper for the extended RealAmplitudes ansatz.

    Parameters
    ----------
    All parameters are passed directly to :func:`real_amplitudes_extended`.
    """

    def __init__(
        self,
        num_qubits: int,
        reps: int = 1,
        entanglement: str | Sequence[Tuple[int, int]] | Callable[[int], Iterable[Tuple[int, int]]] = "full",
        rotation_gate: str = "ry",
        entanglement_gate: str = "cx",
        skip_final_rotation_layer: bool = False,
        skip_final_entanglement_layer: bool = False,
        insert_barriers: bool = False,
        rotation_parameter_prefix: str = "theta",
        entanglement_parameter_prefix: str = "phi",
        share_rotation_parameters: bool = False,
        share_entanglement_parameters: bool = False,
        name: str = "RealAmplitudesExtended",
    ) -> None:
        built = real_amplitudes_extended(
            num_qubits=num_qubits,
            reps=reps,
            entanglement=entanglement,
            rotation_gate=rotation_gate,
            entanglement_gate=entanglement_gate,
            skip_final_rotation_layer=skip_final_rotation_layer,
            skip_final_entanglement_layer=skip_final_entanglement_layer,
            insert_barriers=insert_barriers,
            rotation_parameter_prefix=rotation_parameter_prefix,
            entanglement_parameter_prefix=entanglement_parameter_prefix,
            share_rotation_parameters=share_rotation_parameters,
            share_entanglement_parameters=share_entanglement_parameters,
            name=name,
        )
        super().__init__(built.num_qubits, name=name)
        self.compose(built, inplace=True)

        self.input_params = built.input_params  # type: ignore[attr-defined]
        self.num_rot_layers = built.num_rot_layers  # type: ignore[attr-defined]
        self.num_ent_layers = built.num_ent_layers  # type: ignore[attr-defined]


__all__ = ["RealAmplitudesExtended", "real_amplitudes_extended"]
