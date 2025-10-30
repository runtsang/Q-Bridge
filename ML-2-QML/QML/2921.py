"""Quantum hybrid sampler with attention‑generated parameters.

The quantum circuit is built from the Qiskit SamplerQNN seed but
augments it with an attention‑derived input layer.  The attention
output is treated as a set of rotation parameters for the first
layer of the quantum circuit, while the remaining weights are
trainable.  The circuit is executed on a state‑vector simulator
and the resulting probability distribution is returned.
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler as StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN as QSamplerQNN

import numpy as np

class SamplerQNNGen187:
    """
    Builds a quantum sampler whose first two rotation layers are
    driven by a classical self‑attention module.  The circuit
    architecture follows the original SamplerQNN but replaces the
    static input parameters with a vector that should be supplied
    by the user (e.g. from a classical attention block).
    """
    def __init__(self, attention_output_dim: int = 2, weight_dim: int = 4):
        """
        Parameters
        ----------
        attention_output_dim : int
            Number of parameters coming from the attention mechanism.
            For a 2‑qubit sampler this is typically 2.
        weight_dim : int
            Number of trainable weight parameters.
        """
        self.attention_output_dim = attention_output_dim
        self.weight_dim = weight_dim

        # Parameters
        self.input_params = ParameterVector("input", self.attention_output_dim)
        self.weight_params = ParameterVector("weight", self.weight_dim)

        # Build base circuit
        self.circuit = QuantumCircuit(2)
        # First rotation block driven by attention output
        self.circuit.ry(self.input_params[0], 0)
        self.circuit.ry(self.input_params[1], 1)
        # Entangling layer
        self.circuit.cx(0, 1)
        # Weight driven rotations
        self.circuit.ry(self.weight_params[0], 0)
        self.circuit.ry(self.weight_params[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weight_params[2], 0)
        self.circuit.ry(self.weight_params[3], 1)

        # Sampler primitive
        self.sampler = StatevectorSampler()

        # Wrap in Qiskit Machine Learning SamplerQNN for convenience
        self.qsampler = QSamplerQNN(
            circuit=self.circuit,
            input_params=self.input_params,
            weight_params=self.weight_params,
            sampler=self.sampler
        )

    def run(self, attention_output: np.ndarray, shots: int = 1024) -> dict:
        """
        Execute the circuit with the provided attention output.

        Parameters
        ----------
        attention_output : np.ndarray
            Array of shape (attention_output_dim,) containing the
            rotation angles for the first layer.
        shots : int, optional
            Number of shots for the sampler.

        Returns
        -------
        dict
            Measurement counts (probability distribution) over the
            computational basis states.
        """
        if attention_output.shape[0]!= self.attention_output_dim:
            raise ValueError(f"Expected attention output of shape "
                             f"({self.attention_output_dim},), got "
                             f"{attention_output.shape}")

        # Bind parameters
        param_bindings = {
            self.input_params[0]: attention_output[0],
            self.input_params[1]: attention_output[1]
        }

        # Execute
        result = self.qsampler.run(param_bindings, shots=shots)
        return result

__all__ = ["SamplerQNNGen187"]
