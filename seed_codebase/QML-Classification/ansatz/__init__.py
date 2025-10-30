"""Quantum ansatz model variants available in this package.

The following lightweight builders mirror Qiskit's :class:`RealAmplitudes` ansatz
and common extensions:

* :mod:`.real_amplitudes` – canonical RY layers with CX entanglement.
* :mod:`.real_amplitudes_alternating` – alternating RY/RX rotation layers.
* :mod:`.real_amplitudes_cz` – CZ-based entangling layers.
"""
from .real_amplitudes import RealAmplitudes, real_amplitudes
from .real_amplitudes_alternating import RealAmplitudesAlternating, real_amplitudes_alternating
from .real_amplitudes_cz import RealAmplitudesCZ, real_amplitudes_cz

__all__ = [
    "RealAmplitudes",
    "real_amplitudes",
    "RealAmplitudesAlternating",
    "real_amplitudes_alternating",
    "RealAmplitudesCZ",
    "real_amplitudes_cz",
]
