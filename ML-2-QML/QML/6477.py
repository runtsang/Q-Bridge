import pennylane as qml
from pennylane import numpy as np
from typing import Tuple, Optional

class SamplerQNN:
    """
    Quantum sampler that receives two input parameters and four trainable weights.
    Supports state‑vector and GPU‑accelerated simulators.
    """
    def __init__(self,
                 device_name: str = "default.qubit",
                 device_kwargs: Optional[dict] = None,
                 use_gpu: bool = False) -> None:
        """
        Parameters
        ----------
        device_name : str
            Pennylane device name, e.g. "default.qubit" or "default.mixed".
        device_kwargs : dict, optional
            Additional keyword arguments for the device.
        use_gpu : bool
            When True and a CUDA‑enabled device is available, use it.
        """
        if device_kwargs is None:
            device_kwargs = {}
        if use_gpu:
            try:
                self.dev = qml.device(device_name, wires=2, shots=None, **device_kwargs)
            except Exception:
                self.dev = qml.device("default.qubit", wires=2, shots=None, **device_kwargs)
        else:
            self.dev = qml.device(device_name, wires=2, shots=None, **device_kwargs)

        self.input_shape = (2,)
        self.weight_shape = (4,)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def sample(self,
               inputs: np.ndarray,
               weights: np.ndarray,
               num_shots: int = 1024) -> np.ndarray:
        """
        Draw samples from the variational circuit.

        Parameters
        ----------
        inputs : array_like
            Two input parameters for the RY rotations.
        weights : array_like
            Four trainable weights for the variational layer.
        num_shots : int
            Number of shots to perform.

        Returns
        -------
        probs : np.ndarray
            Estimated probability distribution over the 4 basis states.
        """
        self.dev.shots = num_shots
        probs = self.circuit(inputs, weights)
        self.dev.shots = None
        return probs

    def get_parameter_shapes(self) -> Tuple[Tuple[int,...], Tuple[int,...]]:
        """
        Return the shapes of the input and weight parameters.
        """
        return self.input_shape, self.weight_shape

__all__ = ["SamplerQNN"]
