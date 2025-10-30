import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes, ZFeatureMap
from qiskit.primitives import Sampler, Estimator
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# ------------------------------------------------------------------
# Quantum building blocks inspired by the reference seeds
# ------------------------------------------------------------------


def quantum_fcl(num_qubits: int, shots: int = 1024):
    """Simple parameterised quantum circuit mimicking a fully‑connected layer."""
    backend = Sampler()
    theta = QuantumCircuit.Parameter("theta")
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))
    qc.ry(theta, range(num_qubits))
    qc.measure_all()
    return qc, theta, backend


def quantum_conv_layer(num_qubits: int, param_prefix: str):
    """One‑qubit convolution‑like unit from the QCNN example."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits * 3)
    for i in range(num_qubits // 2):
        idx = 3 * i
        qc.rz(params[idx], 2 * i)
        qc.ry(params[idx + 1], 2 * i + 1)
        qc.cx(2 * i, 2 * i + 1)
    return qc, params


def quantum_pool_layer(num_qubits: int, param_prefix: str):
    """Pooling operation on pairs of qubits."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)
    for i in range(num_qubits // 2):
        idx = 3 * i
        qc.rz(params[idx], 2 * i)
        qc.ry(params[idx + 1], 2 * i + 1)
        qc.cx(2 * i, 2 * i + 1)
    return qc, params


# ------------------------------------------------------------------
# Hybrid quantum autoencoder construction
# ------------------------------------------------------------------


def QuantumHybridAutoencoder(
    latent_dim: int = 3,
    num_trash: int = 2,
    use_qcnn: bool = True,
) -> SamplerQNN:
    """Builds a quantum autoencoder that uses a swap‑test for reconstruction
    and optionally a QCNN‑style convolution ansatz on the latent subspace."""
    algorithm_globals.random_seed = 42
    sampler = Sampler()

    # ---------- Feature encoder ----------
    def encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Apply a RealAmplitudes ansatz to the latent + trash qubits
        ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
        qc.append(ansatz, list(range(num_latent + num_trash)))

        qc.barrier()

        # Swap‑test with auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # ---------- Optional QCNN convolution ansatz ----------
    def qcnn_ansatz(num_qubits):
        qc = QuantumCircuit(num_qubits)
        for i in range(num_qubits // 2):
            conv_qc, _ = quantum_conv_layer(2, f"conv_{i}")
            qc.append(conv_qc, [2 * i, 2 * i + 1])
        return qc

    # ---------- Construct full circuit ----------
    num_latent = latent_dim
    num_trash = num_trash
    circuit = encoder_circuit(num_latent, num_trash)

    # If requested, append a QCNN‑style convolution layer on the latent space
    if use_qcnn:
        conv_qc = qcnn_ansatz(num_latent)
        circuit.append(conv_qc, list(range(num_latent)))

    # ---------- Define QNN ----------
    interpret = lambda x: x  # identity interpret
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],  # no classical feature map
        weight_params=circuit.parameters,
        interpret=interpret,
        output_shape=2,
        sampler=sampler,
    )
    return qnn


# ------------------------------------------------------------------
# Helper that wraps the quantum autoencoder into a hybrid training loop
# ------------------------------------------------------------------


def train_quantum_autoencoder(
    qnn: SamplerQNN,
    data: np.ndarray,
    *,
    epochs: int = 50,
    lr: float = 0.1,
    optimizer_cls=COBYLA,
) -> list[float]:
    """Very light training loop that optimises the circuit parameters."""
    opt = optimizer_cls(maxiter=epochs * 10)
    history = []

    def loss_fn(params):
        # Encode data as a list of parameter values
        results = qnn.run(params)
        preds = results[0]
        # Mean‑squared‑error against target data
        loss = np.mean((preds - data) ** 2)
        return loss

    for epoch in range(epochs):
        params = opt.minimize(loss_fn, qnn.weight_params)
        loss = loss_fn(params)
        history.append(loss)
    return history


__all__ = [
    "QuantumHybridAutoencoder",
    "train_quantum_autoencoder",
]
