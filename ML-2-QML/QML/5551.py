import pennylane as qml
import pennylane.numpy as np
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate

class FraudDetectionHybridQML:
    """
    Quantum back‑end for FraudDetectionHybridML.

    Provides:
      * A qubit variational sampler (SamplerQNN style).
      * A photonic circuit mirroring the original photonic fraud detection.
    """
    def __init__(self, qubit_dev: qml.Device = None) -> None:
        # Default qubit device
        self.qubit_dev = qubit_dev or qml.device("default.qubit", wires=2)

    # ------------------------------------------------------------------
    # Qubit variational sampler
    # ------------------------------------------------------------------
    def qubit_sampler(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Return a 2‑qubit probability distribution from a parameterised circuit.

        Parameters
        ----------
        inputs : np.ndarray
            Two input angles (e.g. from the classical sampler).
        weights : np.ndarray
            Four trainable rotation angles.

        Returns
        -------
        np.ndarray
            Probability vector of shape (4,).
        """
        @qml.qnode(self.qubit_dev, interface="autograd")
        def circuit(a, b, w1, w2, w3, w4):
            qml.RY(a, wires=0)
            qml.RY(b, wires=1)
            qml.CNOT(0, 1)
            qml.RY(w1, wires=0)
            qml.RY(w2, wires=1)
            qml.CNOT(0, 1)
            qml.RY(w3, wires=0)
            qml.RY(w4, wires=1)
            return qml.probs(wires=[0, 1])

        return circuit(*inputs, *weights)

    # ------------------------------------------------------------------
    # Photonic fraud detection circuit
    # ------------------------------------------------------------------
    def photonic_fraud_detection_program(
        self,
        input_params: dict,
        layers: list[dict],
    ) -> sf.Program:
        """
        Build a StrawberryFields program for fraud detection.

        Parameters
        ----------
        input_params : dict
            Parameters for the initial photonic layer.
        layers : list[dict]
            Parameters for subsequent layers (each dict matches FraudLayerParameters).

        Returns
        -------
        sf.Program
            Initialized photonic program ready for execution.
        """
        program = sf.Program(2)

        def _apply_layer(q, params, clip: bool = False):
            BSgate(params["bs_theta"], params["bs_phi"]) | (q[0], q[1])
            for i, phase in enumerate(params["phases"]):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(params["squeeze_r"], params["squeeze_phi"])):
                Sgate(r if not clip else max(-5, min(5, r)), phi) | q[i]
            BSgate(params["bs_theta"], params["bs_phi"]) | (q[0], q[1])
            for i, phase in enumerate(params["phases"]):
                Rgate(phase) | q[i]
            for i, (r, phi) in enumerate(zip(params["displacement_r"], params["displacement_phi"])):
                Dgate(r if not clip else max(-5, min(5, r)), phi) | q[i]
            for i, k in enumerate(params["kerr"]):
                Kgate(k if not clip else max(-1, min(1, k))) | q[i]

        with program.context as q:
            _apply_layer(q, input_params, clip=False)
            for layer in layers:
                _apply_layer(q, layer, clip=True)
        return program

    def run_photonic(
        self,
        program: sf.Program,
        backend: sf.backends.FockSimulator = None,
    ) -> np.ndarray:
        """
        Execute the program and return the photon‑number distribution.

        Parameters
        ----------
        program : sf.Program
        backend : sf.backends.FockSimulator, optional
            If None, a default Fock simulator is used.

        Returns
        -------
        np.ndarray
            Photon‑number distribution over the two modes.
        """
        backend = backend or sf.backends.FockSimulator()
        eng = sf.Engine(backend)
        result = eng.run(program)
        return result.state

__all__ = ["FraudDetectionHybridQML"]
