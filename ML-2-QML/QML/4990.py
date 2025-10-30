from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from qiskit import QuantumCircuit, Parameter, ParameterVector, Statevector
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN

class QuantumKernel:
    """
    A simple quantum kernel implementing the inner product of two
    state‑vectors encoded by a two‑qubit Ry circuit.  The kernel
    returns a real number in [0,1] indicating similarity.
    """
    def __init__(self, circuit: QuantumCircuit | None = None) -> None:
        if circuit is None:
            circ = QuantumCircuit(2)
            circ.ry(Parameter("x0"), 0)
            circ.ry(Parameter("x1"), 1)
            self.circuit = circ
        else:
            self.circuit = circuit

    def __call__(self, x: Sequence[float], y: Sequence[float]) -> float:
        circ_x = self.circuit.copy()
        circ_y = self.circuit.copy()
        circ_x.ry(x[0], 0)
        circ_x.ry(x[1], 1)
        circ_y.ry(y[0], 0)
        circ_y.ry(y[1], 1)
        sv_x = Statevector.from_instruction(circ_x)
        sv_y = Statevector.from_instruction(circ_y)
        return abs(np.vdot(sv_x.data, sv_y.data))

class HybridSamplerQNN:
    """
    Quantum‑centric sampler that uses a parameterized circuit to produce
    a probability distribution over two outcomes.  It also exposes a
    lightweight quantum kernel for similarity evaluation.
    """

    def __init__(self,
                 circuit: QuantumCircuit | None = None,
                 input_params: ParameterVector | None = None,
                 weight_params: ParameterVector | None = None,
                 sampler: StatevectorSampler | None = None) -> None:
        """
        Parameters
        ----------
        circuit:
            The variational circuit.  If omitted, a default 2‑qubit Ry/CX
            circuit is built.
        input_params:
            Parameters that will be bound to input data.  Defaults to a
            two‑parameter vector.
        weight_params:
            Trainable parameters of the circuit.  Defaults to a
            four‑parameter vector.
        sampler:
            Primitive that evaluates the circuit.  If None, a
            :class:`StatevectorSampler` is used.
        """
        if circuit is None:
            inputs2 = ParameterVector("input", 2)
            weights2 = ParameterVector("weight", 4)
            qc2 = QuantumCircuit(2)
            qc2.ry(inputs2[0], 0)
            qc2.ry(inputs2[1], 1)
            qc2.cx(0, 1)
            qc2.ry(weights2[0], 0)
            qc2.ry(weights2[1], 1)
            qc2.cx(0, 1)
            qc2.ry(weights2[2], 0)
            qc2.ry(weights2[3], 1)
            circuit = qc2
        if input_params is None:
            input_params = ParameterVector("input", 2)
        if weight_params is None:
            weight_params = ParameterVector("weight", 4)
        if sampler is None:
            sampler = StatevectorSampler()
        self.sampler_qnn = QiskitSamplerQNN(circuit=circuit,
                                            input_params=input_params,
                                            weight_params=weight_params,
                                            sampler=sampler)
        self.kernel = QuantumKernel(circuit)

    def run(self, inputs: Sequence[float]) -> np.ndarray:
        """
        Evaluate the sampler on a batch of inputs.  ``inputs`` must have
        shape ``(N, 2)``.  Returns a NumPy array of shape ``(N, 2)`` with
        the sampled probabilities.
        """
        return self.sampler_qnn(inputs)

    def kernel_value(self, x: Sequence[float], y: Sequence[float]) -> float:
        """
        Return the quantum kernel similarity between two input vectors.
        """
        return self.kernel(x, y)
