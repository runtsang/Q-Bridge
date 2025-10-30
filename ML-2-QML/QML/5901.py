import numpy as np
from typing import List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN

class Autoencoder__gen303:
    """Quantum autoencoder that uses a RealAmplitudes ansatz and a swap‑test fidelity measurement."""

    def __init__(self, latent_dim: int = 3, trash_dim: int = 2, reps: int = 5, seed: int = 42):
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.reps = reps
        self.seed = seed
        algorithm_globals.random_seed = seed

        self.sampler = Sampler()
        self.circuit = self._build_circuit()
        # No input parameters – the circuit is fully parameterised by the ansatz
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        self.optimizer = COBYLA(maxiter=200, disp=False)

    def _build_circuit(self) -> QuantumCircuit:
        """Build a swap‑test based autoencoder circuit."""
        num_latent = self.latent_dim
        num_trash = self.trash_dim
        n_qubits = num_latent + 2 * num_trash + 1  # +1 for auxiliary qubit
        qr = QuantumRegister(n_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode data into the trash qubits – placeholder: Hadamard on each trash qubit
        for i in range(num_trash):
            qc.h(qr[num_latent + i])

        # Variational ansatz on latent+trash
        ansatz = RealAmplitudes(num_latent + num_trash, reps=self.reps)
        qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

        # Swap‑test with auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(qr[aux])
        for i in range(num_trash):
            qc.cswap(qr[aux], qr[num_latent + i], qr[num_latent + num_trash + i])
        qc.h(qr[aux])
        qc.measure(qr[aux], cr[0])
        return qc

    def _loss_fn(self, params: List[float]) -> float:
        """Compute mean squared error between predicted fidelity and perfect fidelity."""
        # Assign parameters to the circuit
        param_dict = dict(zip(self.circuit.parameters, params))
        self.circuit.assign_parameters(param_dict, inplace=True)
        # Run the circuit and obtain fidelity estimate
        result = self.sampler.run(self.circuit).result()
        probs = result.get_counts()
        # The swap‑test measurement yields 0 (identical) or 1 (orthogonal)
        # We interpret probability of 0 as fidelity
        fidelity = probs.get("0", 0) / sum(probs.values())
        return (1.0 - fidelity) ** 2

    def train(self, *, epochs: int = 5, verbose: bool = False) -> List[float]:
        """Optimise the ansatz parameters to maximise fidelity."""
        history: List[float] = []
        # Initialise parameters randomly
        init_params = np.random.rand(len(self.circuit.parameters))
        params = init_params
        for epoch in range(epochs):
            loss, params = self.optimizer.optimize(params, self._loss_fn)
            history.append(loss)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss:.6f}")
        # Update circuit with the best parameters
        param_dict = dict(zip(self.circuit.parameters, params))
        self.circuit.assign_parameters(param_dict, inplace=True)
        # Rebuild QNN with updated parameters
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )
        return history

    def evaluate(self) -> float:
        """Return the fidelity achieved by the current circuit."""
        result = self.sampler.run(self.circuit).result()
        probs = result.get_counts()
        fidelity = probs.get("0", 0) / sum(probs.values())
        return fidelity

    def get_reconstruction(self) -> Statevector:
        """Return the statevector produced by the circuit (without measurement)."""
        # Remove measurement for statevector simulation
        qc = self.circuit.remove_final_measurements()
        return Statevector.from_instruction(qc)

__all__ = ["Autoencoder__gen303"]
