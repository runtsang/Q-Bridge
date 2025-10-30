import numpy as np
from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler

class SamplerQNN:
    """
    Quantum sampler that emulates the classical architecture while adding
    quantum‑specific features.  The circuit contains:
        * two input Ry rotations (parameterised by the input features)
        * a first CX entangling layer
        * four weight Ry rotations
        * a second CX layer
    An external scale and shift, analogous to the displacement/shift in the
    fraud‑detection analogue, are applied to the measurement probabilities.
    The class exposes a `sample` method that returns a probability distribution
    over the four computational basis states.
    """

    def __init__(self,
                 clip_value: float = 5.0,
                 seed: int | None = None):
        self.clip_value = clip_value
        if seed is not None:
            np.random.seed(seed)

        # Parameters
        self.input_params = ParameterVector('in', length=2)
        self.weight_params = ParameterVector('w', length=4)
        self.scale_params = ParameterVector('scale', length=2)
        self.shift_params = ParameterVector('shift', length=2)

        self.circuit = self._build_circuit()
        self.sampler = StatevectorSampler()

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        # Input rotations
        for i, p in enumerate(self.input_params):
            qc.ry(p, i)
        # Entangling layers
        qc.cx(0, 1)
        for i, p in enumerate(self.weight_params):
            qc.ry(p, i)
        qc.cx(0, 1)
        return qc

    def _postprocess(self, probs: np.ndarray) -> np.ndarray:
        # Clip scale and shift to avoid extreme values
        scale = np.clip(np.asarray(self.scale_params), -self.clip_value, self.clip_value)
        shift = np.clip(np.asarray(self.shift_params), -self.clip_value, self.clip_value)
        probs = probs * scale + shift
        probs = np.maximum(probs, 0)
        probs = probs / np.sum(probs, axis=-1, keepdims=True)
        return probs

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Sample from the quantum circuit for a batch of 2‑dimensional inputs.

        Parameters
        ----------
        inputs : np.ndarray
            Shape (batch_size, 2).  Each row contains the two input features.

        Returns
        -------
        probs : np.ndarray
            Shape (batch_size, 4) containing the probability distribution
            over computational basis states after post‑processing.
        """
        batch_size = inputs.shape[0]
        probs = np.zeros((batch_size, 4))

        for idx, inp in enumerate(inputs):
            # Bind input parameters
            bound = dict(zip(self.input_params, inp))
            # Randomly initialise weights, scale and shift within bounds
            for p in self.weight_params:
                bound[p] = np.random.uniform(-self.clip_value, self.clip_value)
            for p in self.scale_params:
                bound[p] = np.random.uniform(-self.clip_value, self.clip_value)
            for p in self.shift_params:
                bound[p] = np.random.uniform(-self.clip_value, self.shift_value)

            # Get statevector from the sampler
            state = self.sampler.run(self.circuit, bound).result().get_statevector()
            probs[idx] = self._postprocess(np.abs(state)**2)

        return probs

    def reset_parameters(self) -> None:
        """
        Randomly reinitialise all parameters within the clipping range.
        """
        all_params = list(self.input_params) + list(self.weight_params) + \
                     list(self.scale_params) + list(self.shift_params)
        for p in all_params:
            val = np.random.uniform(-self.clip_value, self.clip_value)
            self.circuit.assign_parameters({p: val}, inplace=True)

__all__ = ["SamplerQNN"]
