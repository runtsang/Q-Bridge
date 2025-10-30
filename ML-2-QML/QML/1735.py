import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

def Autoencoder__gen320(device: str = "qasm_simulator", shots: int = 8192) -> SamplerQNN:
    """
    Quantum autoencoder using a parameter‑shared RealAmplitudes ansatz and a swap‑test measurement.
    The circuit is device‑aware: it defaults to a QASM simulator but can be overridden.
    """
    algorithm_globals.random_seed = 42
    sampler = Sampler(device=device, shots=shots)

    def ansatz(num_qubits: int, reps: int = 3) -> QuantumCircuit:
        """Parameter‑shared ansatz across all qubits."""
        qc = QuantumCircuit(num_qubits)
        for _ in range(reps):
            qc.append(RealAmplitudes(num_qubits, reps=1), range(num_qubits))
        return qc

    def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
        """Build the variational encoder with a swap‑test for fidelity."""
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qr, cr)

        # Encoder ansatz
        qc.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        qc.barrier()

        # Swap‑test with auxiliary qubit
        aux = num_latent + 2 * num_trash
        qc.h(aux)
        for i in range(num_trash):
            qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
        qc.h(aux)
        qc.measure(aux, cr[0])
        return qc

    # Hyper‑parameters
    num_latent = 3
    num_trash = 2
    circuit = autoencoder_circuit(num_latent, num_trash)

    # Interpret measurement as fidelity estimate (0 → bad, 1 → good)
    def fidelity_interpret(x: np.ndarray) -> float:
        return float(1 - np.mean(x))  # 1 - probability of |0⟩ gives reconstruction error

    # Build SamplerQNN
    qnn = SamplerQNN(
        circuit=circuit,
        input_params=[],          # No classical inputs for this simple demo
        weight_params=circuit.parameters,
        interpret=fidelity_interpret,
        output_shape=1,
        sampler=sampler,
    )
    return qnn

def train_qml_autoencoder(qnn: SamplerQNN, *, epochs: int = 50, lr: float = 0.01):
    """
    Simple training loop for the quantum autoencoder.
    Uses COBYLA optimizer to minimize reconstruction error.
    """
    optimizer = COBYLA()
    for _ in range(epochs):
        # COBYLA expects a callable that returns a scalar loss
        def loss_fn(params):
            qnn.set_weights(params)
            scores = qnn.predict([[]])[0]
            return 1 - np.mean(scores)  # minimize error

        result = optimizer.optimize(num_vars=len(qnn.parameters), objective_function=loss_fn, initial_point=np.random.randn(len(qnn.parameters)))
        qnn.set_weights(result.x)
    return qnn

__all__ = ["Autoencoder__gen320", "train_qml_autoencoder"]
