"""Hybrid quantum auto‑encoder with embedded regression estimator.

The circuit is a concatenation of a RealAmplitudes encoder, a swap‑test
reconstruction block, and a lightweight EstimatorQNN that predicts a
scalar target from the latent subspace.  The implementation follows
the patterns of the original quantum Autoencoder and EstimatorQNN
examples while exposing a single forward method that returns both
reconstruction probabilities and a regression estimate.
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.utils import algorithm_globals

algorithm_globals.random_seed = 42

def _build_autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Constructs the encoder + swap‑test circuit that produces a latent
    subspace and a reconstruction measurement on an auxiliary qubit.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Encoder: RealAmplitudes ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap‑test between latent and trash qubits
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])

    return qc

def _build_estimator_circuit() -> tuple[QuantumCircuit, list[Parameter]]:
    """Simple two‑parameter single‑qubit regression circuit."""
    params = [Parameter("input1"), Parameter("weight1")]
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(params[0], 0)
    qc.rx(params[1], 0)
    return qc, params

class HybridAutoEncoder:
    """Quantum hybrid auto‑encoder with embedded regression estimator."""
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 3,
                 num_trash: int = 2,
                 estimator_weight_init: float = 0.1
                 ) -> None:
        self.auto_circuit = _build_autoencoder_circuit(latent_dim, num_trash)
        self.estim_circuit, self.estim_params = _build_estimator_circuit()

        # Combine circuits: estimator will act on the first qubit of the autoencoder
        # For simplicity we keep them separate but allow joint execution.
        self.combined_circuit = self.auto_circuit.compose(
            self.estim_circuit, front=False
        )

        # Sampler for reconstruction proxy
        self.sampler = StatevectorSampler()
        self.auto_qnn = SamplerQNN(
            circuit=self.auto_circuit,
            input_params=[],
            weight_params=self.auto_circuit.parameters,
            interpret=lambda sv: np.real(sv.data[0]),
            output_shape=(input_dim,)
        )

        # Estimator for regression
        observable = SparsePauliOp.from_list([("X", 1)])
        self.estimator = StatevectorEstimator()
        self.estim_qnn = EstimatorQNN(
            circuit=self.estim_circuit,
            observables=observable,
            input_params=[self.estim_params[0]],
            weight_params=[self.estim_params[1]],
            estimator=self.estimator
        )

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Run the autoencoder circuit and return the measurement probability of the auxiliary qubit."""
        result = self.sampler.run(self.auto_circuit, shots=1024)
        probs = result.data[0]
        return probs

    def predict(self, latent: np.ndarray) -> float:
        """Perform regression using the estimator circuit bound to a latent value."""
        bound_circuit = self.estim_circuit.bind_parameters(
            {self.estim_params[0]: latent}
        )
        result = self.estimator.run(bound_circuit, shots=1024)
        return result.data[0].real

    def forward(self, inputs: np.ndarray) -> tuple[np.ndarray, float]:
        """Return reconstruction proxy and regression prediction."""
        recon = self.encode(inputs)
        # Use a simple function of recon as latent; here we use its mean
        latent = np.mean(recon)
        pred = self.predict(latent)
        return recon, pred

__all__ = ["HybridAutoEncoder"]
