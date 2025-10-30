"""Quantum sampler network that mirrors the hybrid classical architecture.

This module builds a parameterised 2‑qubit circuit that accepts both
input angles and weight angles.  It uses Qiskit's
``StatevectorSampler`` to evaluate the expectation value of the
``ry`` rotations, emulating the behaviour of the classical ``run``
method.  The circuit is compatible with Qiskit Machine Learning's
``SamplerQNN`` wrapper, which provides convenient integration with
the quantum SDK.

Key features
------------
* Two sets of parameters: ``inputs`` (2 angles) and ``weights`` (4 angles).
* A simple entangling pattern: RY → CX → RY → CX → RY → RY.
* Optional ``sample`` method that returns raw measurement counts.
* ``expectation`` method that returns the weighted sum of states,
  matching the classical tanh‑activated linear expectation.

Example
-------
>>> from SamplerQNN__gen018 import SamplerQNN__gen018 as QSampler
>>> qsampler = QSampler()
>>> thetas = [0.1, 0.5, 0.9, 0.2]
>>> print(qsampler.expectation(thetas))
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, List

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler


class SamplerQNN__gen018:
    """
    Quantum sampler network that encapsulates a parameterised circuit
    and provides both sampling and expectation interfaces.

    Parameters
    ----------
    backend : str, optional
        Backend name; defaults to AerSimulator.
    shots : int, optional
        Number of shots for sampling; defaults to 1024.
    """

    def __init__(self, backend: str = "aer_simulator", shots: int = 1024) -> None:
        # Parameter vectors for inputs (2) and weights (4).
        self.inputs = ParameterVector("input", 2)
        self.weights = ParameterVector("weight", 4)

        # Build the circuit.
        self.circuit = QuantumCircuit(2)
        # Input rotations.
        self.circuit.ry(self.inputs[0], 0)
        self.circuit.ry(self.inputs[1], 1)
        # Entanglement.
        self.circuit.cx(0, 1)
        # Weight rotations.
        self.circuit.ry(self.weights[0], 0)
        self.circuit.ry(self.weights[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights[2], 0)
        self.circuit.ry(self.weights[3], 1)

        # Sampler primitive.
        self.sampler = StatevectorSampler(backend=AerSimulator())

        # Wrap with Qiskit Machine Learning SamplerQNN for convenience.
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs,
            weight_params=self.weights,
            sampler=self.sampler,
        )

    # ------------------------------------------------------------------ #
    #  Sampling interface
    # ------------------------------------------------------------------ #
    def sample(self, thetas: Iterable[float]) -> List[int]:
        """
        Execute the circuit with the supplied weight parameters and return
        raw measurement counts.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of 4 weight angles.

        Returns
        -------
        List[int]
            List of counts for each measurement outcome.
        """
        if len(thetas)!= 4:
            raise ValueError("Expected 4 weight parameters.")
        counts = self.sampler_qnn.run(
            input_vals=[0.0, 0.0],  # fixed zero inputs for sampling
            weight_vals=list(thetas),
        )
        return counts

    # ------------------------------------------------------------------ #
    #  Expectation interface
    # ------------------------------------------------------------------ #
    def expectation(self, thetas: Iterable[float]) -> float:
        """
        Compute the weighted sum of measurement outcomes, analogous to the
        classical tanh‑activated linear expectation.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of 4 weight angles.

        Returns
        -------
        float
            Expectation value.
        """
        counts = self.sample(thetas)
        total_shots = sum(counts)
        if total_shots == 0:
            return 0.0
        expectation = 0.0
        for outcome, count in zip(counts, [0, 1, 2, 3]):  # 00, 01, 10, 11
            expectation += count / total_shots * float(outcome)
        return expectation

    # ------------------------------------------------------------------ #
    #  Utility: draw the circuit
    # ------------------------------------------------------------------ #
    def draw(self, style: str = "mpl") -> None:
        """
        Draw the underlying quantum circuit.

        Parameters
        ----------
        style : str, optional
            Drawing style; defaults to "mpl".
        """
        print(self.circuit.draw(style=style))

__all__ = ["SamplerQNN__gen018"]
