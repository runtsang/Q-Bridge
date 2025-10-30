import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN

class HybridAutoencoder:
    """
    Quantum decoder for the hybrid autoencoder.

    The circuit applies a parameterised Ry rotation on each qubit
    and measures all qubits.  The probabilities of measuring |0>
    on each qubit are interpreted as the reconstructed input.
    """
    def __init__(self, input_dim: int, latent_dim: int | None = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim or input_dim
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=self._interpret,
            output_shape=self.input_dim,
            sampler=StatevectorSampler(),
        )

    def _build_circuit(self) -> QuantumCircuit:
        qr = QuantumRegister(self.input_dim)
        cr = ClassicalRegister(self.input_dim)
        circuit = QuantumCircuit(qr, cr)
        # Parameterised Ry rotations on each qubit
        for i in range(self.input_dim):
            circuit.ry(0, i)  # placeholder, will be replaced by parameters
        circuit.measure(qr, cr)
        return circuit

    def _interpret(self, bitstrings: np.ndarray) -> np.ndarray:
        # bitstrings shape (N, input_dim) of 0/1
        # return probability of 0 for each qubit
        return 1.0 - bitstrings.mean(axis=0)

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        """
        Forward pass of the quantum decoder.

        Parameters
        ----------
        latent : np.ndarray
            Latent vector of shape (latent_dim,) or (batch, latent_dim).

        Returns
        -------
        np.ndarray
            Reconstructed input of shape (input_dim,) or (batch, input_dim).
        """
        if latent.ndim == 1:
            latent = latent.reshape(1, -1)
        outputs = []
        for sample in latent:
            # Assign the sample to the circuit parameters
            param_dict = {name: val for name, val in zip(self.circuit.parameters, sample)}
            self.circuit.assign_parameters(param_dict, inplace=True)
            out = self.qnn(sample)  # sample is not used as input_params
            outputs.append(out)
        return np.array(outputs).reshape(-1, self.input_dim)

    def train(self, data: np.ndarray, epochs: int = 100, lr: float = 0.01) -> None:
        """
        Train the quantum decoder to minimise MSE between its output and the data.

        Parameters
        ----------
        data : np.ndarray
            Training data of shape (N, input_dim).
        epochs : int
            Number of optimisation iterations.
        lr : float
            Learning rate (used by the optimizer).
        """
        from qiskit_machine_learning.optimizers import COBYLA
        opt = COBYLA(maxiter=epochs)

        def loss(params: np.ndarray) -> float:
            # assign parameters to the circuit
            for name, val in zip(self.circuit.parameters, params):
                self.circuit.assign_parameters({name: val}, inplace=True)
            preds = self.__call__(data)
            return ((preds - data) ** 2).mean()

        opt.optimize(num_vars=len(self.circuit.parameters), objective_function=loss)
