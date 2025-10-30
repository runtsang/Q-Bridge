import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from typing import List

class AutoencoderGen337:
    """
    Quantum auto‑encoder implemented as a parameter‑efficient sampler QNN.
    The circuit consists of a RealAmplitudes ansatz, a swap‑test style
    auxiliary qubit, and optional domain‑wall encoding.
    """
    def __init__(
        self,
        num_latent: int,
        num_trash: int = 2,
        ansatz_reps: int = 5,
        cost_fidelity: bool = True,
    ):
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.ansatz_reps = ansatz_reps
        self.cost_fidelity = cost_fidelity

        algorithm_globals.random_seed = 42
        self.sampler = Sampler()

        # Build the quantum circuit
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=(2,),
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        # Total qubits: latent + 2*trash + 1 auxiliary
        total_q = self.num_latent + 2 * self.num_trash + 1
        qr = QuantumRegister(total_q, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encode latent and trash qubits using RealAmplitudes ansatz
        ansatz = RealAmplitudes(self.num_latent + self.num_trash, reps=self.ansatz_reps)
        qc.compose(ansatz, range(0, self.num_latent + self.num_trash), inplace=True)

        # Swap‑test style auxiliary qubit
        aux = self.num_latent + 2 * self.num_trash
        qc.h(aux)
        for i in range(self.num_trash):
            qc.cswap(aux, self.num_latent + i, self.num_latent + self.num_trash + i)
        qc.h(aux)

        qc.measure(aux, cr[0])
        return qc

    def fidelity_cost(self, parameters: np.ndarray) -> float:
        """
        Evaluate the circuit with given parameters and compute
        the fidelity with respect to the target |0⟩ auxiliary state.
        """
        # Bind parameters and run sampler
        bound_circuit = self.circuit.bind_parameters(parameters)
        result = self.sampler.run(bound_circuit).result()
        counts = result.get_counts(bound_circuit)
        # Compute probability of measuring '0' on the auxiliary qubit
        prob_zero = counts.get('0', 0) / sum(counts.values()) if counts else 0.0
        # Return negative fidelity for minimisation
        return -prob_zero

    def train(self, epochs: int = 50, maxiter: int = 500) -> List[float]:
        """
        Simple gradient‑free optimiser (COBYLA) to minimise the fidelity
        cost. Returns the cost history.
        """
        opt = COBYLA(maxiter=maxiter)
        cost_history: List[float] = []

        def objective(params: np.ndarray) -> float:
            cost = self.fidelity_cost(params)
            cost_history.append(cost)
            return cost

        # Random initial parameters
        init_params = np.random.rand(len(self.circuit.parameters))
        opt.minimize(objective, init_params)
        return cost_history

__all__ = ["AutoencoderGen337"]
