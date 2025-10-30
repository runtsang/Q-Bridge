import json
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, PauliEvolutionGate
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ------------------------------------------------------------------
# Variational auto‑encoder circuit
# ------------------------------------------------------------------
def auto_encoder_qc(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Return a quantum circuit that encodes `num_latent` logical qubits
    and uses `num_trash` ancillary qubits for a swap‑test style measurement.
    The circuit is inspired by the original QML Autoencoder demo and
    extends it with a RealAmplitudes ansatz.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    # variational ansatz on the logical and trash qubits
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    qc.append(ansatz, range(0, num_latent + num_trash))
    qc.barrier()

    # swap‑test with auxiliary qubit
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

# ------------------------------------------------------------------
# Domain‑wall preparation (used in hybrid conv‑quantum example)
# ------------------------------------------------------------------
def domain_wall(circuit: QuantumCircuit, a: int, b: int) -> QuantumCircuit:
    """Place X gates on all qubits in range [int(b/2), b)."""
    for i in np.arange(int(b / 2), int(b)):
        circuit.x(i)
    return circuit

# ------------------------------------------------------------------
# Sampler‑based quantum neural network
# ------------------------------------------------------------------
def build_qnn(num_latent: int, num_trash: int) -> "SamplerQNN":
    """
    Build a SamplerQNN that uses the auto‑encoder circuit as a variational layer.
    The weight parameters are the circuit parameters; the input parameters are empty
    because all classical data is encoded into the circuit ansatz.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()
    qc = auto_encoder_qc(num_latent, num_trash)
    # wrap with domain‑wall to test qubit ordering
    dwc = domain_wall(QuantumCircuit(5), 0, 5)
    qc.compose(dwc, range(num_latent + num_trash), inplace=True)

    def identity_interpret(x: np.ndarray) -> np.ndarray:
        """Return the raw measurement probabilities."""
        return x

    from qiskit_machine_learning.neural_networks import SamplerQNN
    qnn = SamplerQNN(
        circuit=qc,
        input_params=[],
        weight_params=qc.parameters,
        interpret=identity_interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    # build a hybrid quantum neural network
    qnn = build_qnn(num_latent=3, num_trash=2)

    # run a dummy forward pass
    dummy_input = np.random.rand(1, 2)
    # SamplerQNN expects a dict of input_params -> values; here empty
    outputs = qnn.forward({})
    print("Quantum neural network output:", outputs)

    # plot the circuit
    qnn.circuit.draw(output="mpl", style="clifford")
    plt.show()
