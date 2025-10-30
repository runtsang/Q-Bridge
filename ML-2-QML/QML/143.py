from __future__ import annotations

import pennylane as qml
import pennylane.numpy as np
from pennylane import qnn
from pennylane import default_qubit

__all__ = ["SamplerQNN"]


def SamplerQNN() -> qnn.SamplerQNN:
    """
    Builds a simple 2‑qubit variational sampler.
    The circuit contains:
      • Two input rotations Ry(θ_i) on each qubit.
      • A single entangling layer (CNOT).
      • Two layers of trainable Ry rotations.
    The resulting SamplerQNN can be used as a differentiable quantum neural
    network that returns measurement samples.
    """

    dev = default_qubit.Device("default.qubit", wires=2)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(inputs, weights):
        # Input rotations
        qml.Ry(inputs[0], wires=0)
        qml.Ry(inputs[1], wires=1)
        # Entangling
        qml.CNOT(wires=[0, 1])

        # Trainable rotations
        qml.Ry(weights[0], wires=0)
        qml.Ry(weights[1], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.Ry(weights[2], wires=0)
        qml.Ry(weights[3], wires=1)

        return qml.measure()

    # Wrap the circuit in a SamplerQNN that emits samples from the measurement.
    sampler_qnn = qnn.SamplerQNN(
        circuit=circuit,
        input_params=[0, 1],      # indices of input parameters in the qnode
        weight_params=[0, 1, 2, 3],  # indices of weight parameters
    )
    return sampler_qnn
