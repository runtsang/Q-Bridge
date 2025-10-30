"""AutoencoderGen048: a quantum autoencoder with a variational circuit and domain‑wall injection.

The implementation builds a parameterised quantum circuit that compresses a quantum state
into a smaller latent register. The circuit uses a RealAmplitudes ansatz, a domain‑wall
pre‑processing step, and a swap‑test based measurement to compute the fidelity between
encoded and target states. Training uses a parameter‑shift gradient with the COBYLA optimizer.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, X
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

class AutoencoderGen048:
    """Quantum autoencoder that compresses a state into a latent register."""

    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        reps: int = 3,
        backend: str | None = None,
        seed: int = 42,
    ) -> None:
        algorithm_globals.random_seed = seed
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.reps = reps
        self.backend = backend or "qasm_simulator"
        self.sampler = Sampler()
        self._build_circuit()

    def _domain_wall(self, qc: QuantumCircuit) -> QuantumCircuit:
        """Inject a domain wall into the trash qubits."""
        for i in range(self.num_trash):
            qc.x(i)
        return qc

    def _build_circuit(self) -> None:
        total_qubits = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Ansatz on the latent + one trash qubit
        ansatz = RealAmplitudes(total_qubits - 1, reps=self.reps)
        qc.compose(ansatz, list(range(total_qubits - 1)), inplace=True)

        # Domain wall on the trash qubits
        qc = self._domain_wall(qc)

        # Swap‑test for fidelity estimation
        aux = total_qubits - 1
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])

        self.circuit = qc

    def encode(self, params: np.ndarray) -> Statevector:
        """Return the encoded statevector for a given set of circuit parameters."""
        bound_qc = self.circuit.bind_parameters(params)
        return Statevector.from_instruction(bound_qc)

    def loss(self, params: np.ndarray, target: Statevector) -> float:
        """Fidelity‑based loss: 1 - |⟨ψ_target|ψ_encoded⟩|²."""
        encoded = self.encode(params)
        fidelity = np.abs(np.vdot(target.data, encoded.data)) ** 2
        return 1.0 - fidelity

    def train(
        self,
        target: Statevector,
        initial_params: np.ndarray | None = None,
        max_iter: int = 200,
        tol: float = 1e-3,
    ) -> np.ndarray:
        """Train the autoencoder to match the target state.

        Parameters
        ----------
        target : Statevector
            The quantum state to be compressed.
        initial_params : np.ndarray | None
            Optional initial parameters; otherwise random.
        max_iter : int
            Maximum number of COBYLA iterations.
        tol : float
            Convergence tolerance.

        Returns
        -------
        params : np.ndarray
            Optimised circuit parameters.
        """
        if initial_params is None:
            initial_params = np.random.randn(self.circuit.num_parameters)
        optimizer = COBYLA(maxiter=max_iter, tol=tol, disp=False)
        result = optimizer.optimize(num_vars=len(initial_params), objective_function=self.loss, initial_point=initial_params, args=(target,))
        return result[0]

    def decode(self, params: np.ndarray, latent_state: Statevector) -> Statevector:
        """Reconstruct the full state from the latent register."""
        # Prepare latent state on the first num_latent qubits
        prep = QuantumCircuit(self.num_latent)
        prep.initialize(latent_state.data, range(self.num_latent))
        # Append rest of the circuit
        full_qc = prep.compose(self.circuit, front=True)
        return Statevector.from_instruction(full_qc)

    def visualize(self) -> None:
        """Print the circuit diagram."""
        print(self.circuit.draw(visualization=True, style="clifford"))

    def fidelity(self, params: np.ndarray, target: Statevector) -> float:
        """Return the fidelity between encoded and target states."""
        encoded = self.encode(params)
        return np.abs(np.vdot(target.data, encoded.data)) ** 2

__all__ = ["AutoencoderGen048"]
