"""Quantum sampler network with a deeper variational circuit.

The function builds a 2‑qubit variational circuit with three entangling layers,
parameterised by 6 input angles and 6 weight angles. It uses the QASM sampler
to produce discrete samples, which is more realistic for noisy hardware.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import QasmSampler


def SamplerQNNGen346() -> SamplerQNN:
    """
    Construct a 2‑qubit variational sampler.

    Parameters
    ----------
    inputs2 : ParameterVector
        Two input rotation angles.
    weights2 : ParameterVector
        Six weight rotation angles.

    Returns
    -------
    SamplerQNN
        A Qiskit Machine Learning SamplerQNN instance ready for training.
    """
    inputs2 = ParameterVector("input", 2)
    weights2 = ParameterVector("weight", 6)

    qc = QuantumCircuit(2)

    # Layer 1: input rotations
    qc.ry(inputs2[0], 0)
    qc.ry(inputs2[1], 1)

    # Layer 2: entanglement + weight rotations
    for i in range(3):
        qc.cx(0, 1)
        qc.ry(weights2[2 * i], 0)
        qc.ry(weights2[2 * i + 1], 1)

    # Final measurement to produce samples
    qc.measure_all()

    sampler = QasmSampler(shots=8192)

    sampler_qnn = SamplerQNN(
        circuit=qc,
        input_params=inputs2,
        weight_params=weights2,
        sampler=sampler,
    )
    return sampler_qnn
