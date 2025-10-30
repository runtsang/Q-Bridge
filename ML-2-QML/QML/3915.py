import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from dataclasses import dataclass
from typing import Tuple

# --------------------------------------------------------------------------- #
# Quantum autoencoder circuit
# --------------------------------------------------------------------------- #

@dataclass
class QuantumAutoencoderConfig:
    """
    Configuration for the quantum autoencoder.

    Parameters
    ----------
    num_latent : int, default=3
        Number of qubits used to encode the latent representation.
    num_trash : int, default=2
        Number of ancillary qubits used for the swap‑test.
    reps : int, default=5
        Number of repetitions of the ansatz layers.
    """
    num_latent: int = 3
    num_trash: int = 2
    reps: int = 5


class AutoencoderHybrid:
    """
    Quantum autoencoder that implements a variational encoder and a
    swap‑test‑based decoder.  The circuit is constructed from a
    RealAmplitudes ansatz followed by a swap test that compares the
    encoded state with an auxiliary qubit.  The output is a 2‑dimensional
    probability vector that can be interpreted as the reconstructed
    input.

    The class exposes a :class:`qiskit_machine_learning.neural_networks.SamplerQNN`
    instance that can be used in a hybrid training loop with a classical
    optimiser.
    """

    def __init__(self, config: QuantumAutoencoderConfig) -> None:
        self.config = config
        algorithm_globals.random_seed = 42
        self.sampler = StatevectorSampler()
        self.circuit = self._build_circuit()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,
            output_shape=2,
            sampler=self.sampler,
        )

    def _build_circuit(self) -> QuantumCircuit:
        num_latent = self.config.num_latent
        num_trash = self.config.num_trash
        total_qubits = num_latent + 2 * num_trash + 1  # +1 auxiliary for swap test

        qr = QuantumRegister(total_qubits, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)

        # Encode: RealAmplitudes ansatz over latent+trash qubits
        circuit.append(
            RealAmplitudes(num_latent + num_trash, reps=self.config.reps),
            range(num_latent + num_trash),
        )

        circuit.barrier()

        # Swap test with auxiliary qubit
        aux_idx = num_latent + 2 * num_trash
        circuit.h(aux_idx)
        for i in range(num_trash):
            circuit.cswap(aux_idx, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux_idx)
        circuit.measure(aux_idx, cr[0])

        return circuit

    def get_qnn(self):
        """Return the underlying SamplerQNN."""
        return self.qnn

    def sample(self, num_shots: int = 1024):
        """Run the circuit on the sampler backend and return measurement counts."""
        return self.sampler.run(self.circuit, shots=num_shots).result().get_counts()

    def circuit_draw(self, filename: str | None = None):
        """Generate a matplotlib drawing of the circuit."""
        return self.circuit.draw(output="mpl", style="clifford", filename=filename)


# --------------------------------------------------------------------------- #
# Utility: hybrid optimisation example
# --------------------------------------------------------------------------- #

def hybrid_optimization_example():
    """
    Demonstrates a simple hybrid optimisation loop that jointly trains
    the quantum autoencoder with a classical optimiser.  The objective
    is to minimise the mean‑squared error between the desired target
    vectors and the QNN output.

    This function is *illustrative* and can be adapted to use a
    deeper classical encoder/decoder if desired.
    """
    config = QuantumAutoencoderConfig(num_latent=3, num_trash=2, reps=5)
    qa = AutoencoderHybrid(config)

    # Dummy target data: 2‑dimensional vectors
    targets = np.array([[0.8, 0.2], [0.5, 0.5], [0.1, 0.9]])
    optimizer = COBYLA(maxiter=200)

    def loss_fn(params):
        # Update QNN parameters
        qa.qnn.set_weights(params)
        # Run QNN for each target
        preds = []
        for _ in targets:
            res = qa.qnn.forward(np.zeros(0))
            preds.append(res)
        preds = np.array(preds)
        return np.mean((preds - targets) ** 2)

    # Initialise parameters and optimise
    init_params = np.random.randn(len(qa.qnn.weight_params))
    opt_params = optimizer.optimize(loss_fn, init_params)
    qa.qnn.set_weights(opt_params)
    print("Optimised parameters:", opt_params)

__all__ = [
    "AutoencoderHybrid",
    "QuantumAutoencoderConfig",
    "hybrid_optimization_example",
]
