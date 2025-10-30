import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes, PauliFeatureMap
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.primitives import Sampler

def QuantumAutoencoder(
    latent_dim: int = 32,
    depth: int = 2,
    amplitude_encoding: bool = True
) -> SamplerQNN:
    """
    Returns a SamplerQNN that maps a classical latent vector to a quantum state
    using a feature map followed by a parameterised RealAmplitudes ansatz.
    The circuit is configured for a swap‑test style fidelity estimation.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # Feature map: amplitude encoding of the latent vector
    if amplitude_encoding:
        # Use PauliFeatureMap to embed the classical vector
        feature_map = PauliFeatureMap(
            num_qubits=latent_dim,
            reps=1,
            paulis="Z",
            insert_barriers=True,
        )
    else:
        # Simple RY rotations for each qubit
        feature_map = QuantumCircuit(latent_dim)
        for i in range(latent_dim):
            feature_map.ry(0, i)  # placeholder; to be replaced by input_params

    # Parameterised ansatz
    ansatz = RealAmplitudes(num_qubits=latent_dim, reps=depth)

    # Full circuit: feature map followed by ansatz
    qc = QuantumCircuit(latent_dim)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)

    # Swap test with an auxiliary qubit for fidelity measurement
    aux_q = QuantumRegister(1, "aux")
    qc.add_register(aux_q)
    qc.h(aux_q[0])
    for i in range(latent_dim):
        qc.cswap(aux_q[0], i, i)  # trivial swap; placeholder for actual logic
    qc.h(aux_q[0])
    cr = ClassicalRegister(1, "c")
    qc.add_register(cr)
    qc.measure(aux_q[0], cr[0])

    # Interpret the measurement: probability of |0> gives fidelity proxy
    def interpret(x: np.ndarray) -> float:
        return x[0]  # probability of measuring 0

    qnn = SamplerQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        interpret=interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn
