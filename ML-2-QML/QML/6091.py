"""Quantum feature generators for HybridEstimatorQNN.

The module exposes two builder functions that return a Qiskit QuantumCircuit
or a Strawberry Fields Program.  Both circuits are parameterised by a
dictionary of real values that are clipped to maintain numerical stability.
These functions are used by the hybrid estimator to extract quantum
features during training.

The builders are intentionally lightweight; they are called for every forward
pass to keep the interface simple.  In a production setting one could cache
the circuits and reuse them.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter as QiskitParameter
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

# --------------------------------------------------------------------------- #
# Parameter clipping helper
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return float(max(-bound, min(bound, value)))

# --------------------------------------------------------------------------- #
# Qiskit circuit builder
# --------------------------------------------------------------------------- #
def build_qubit_circuit(params: dict) -> QuantumCircuit:
    """
    Build a 2‑qubit feature circuit.

    Parameters
    ----------
    params
        Dictionary with keys:
        - ``theta`` : beam‑splitter rotation angle
        - ``phi``   : beam‑splitter phase
        - ``ry``    : rotation around Y
        - ``rx``    : rotation around X

    Returns
    -------
    QuantumCircuit
        The constructed circuit ready for simulation or execution.
    """
    theta = _clip(params["theta"], np.pi)
    phi   = _clip(params["phi"], np.pi)
    ry    = _clip(params["ry"], np.pi)
    rx    = _clip(params["rx"], np.pi)

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.ry(ry, 0)
    qc.rx(rx, 0)
    qc.cx(0, 1)
    qc.h(1)
    return qc

# --------------------------------------------------------------------------- #
# Strawberry Fields program builder
# --------------------------------------------------------------------------- #
def build_photonic_program(params: dict) -> sf.Program:
    """
    Build a 2‑mode photonic circuit.

    Parameters
    ----------
    params
        Dictionary containing:
        - ``bs_theta``, ``bs_phi``
        - ``phases``          : tuple[float, float]
        - ``squeeze_r``       : tuple[float, float]
        - ``squeeze_phi``     : tuple[float, float]
        - ``displacement_r``  : tuple[float, float]
        - ``displacement_phi``: tuple[float, float]
        - ``kerr``            : tuple[float, float]
    """
    program = sf.Program(2)
    with program.context as q:
        BSgate(_clip(params["bs_theta"], np.pi), _clip(params["bs_phi"], np.pi)) | (q[0], q[1])
        for i, phase in enumerate(params["phases"]):
            Rgate(phase) | q[i]
        for r, phi in zip(params["squeeze_r"], params["squeeze_phi"]):
            Sgate(_clip(r, 5), phi) | q[0]
        for r, phi in zip(params["displacement_r"], params["displacement_phi"]):
            Dgate(_clip(r, 5), phi) | q[1]
        for k in params["kerr"]:
            Kgate(_clip(k, 1)) | q[0]
    return program

__all__ = ["build_qubit_circuit", "build_photonic_program"]
