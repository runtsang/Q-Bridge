import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from qiskit.utils import algorithm_globals

def _random_seed(seed: int = 42) -> None:
    algorithm_globals.random_seed = seed

def auto_encoder_circuit(num_latent: int, num_trash: int, reps: int = 5) -> QuantumCircuit:
    """Builds the core auto‑encoder circuit with a RealAmplitudes ansatz."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.compose(RealAmplitudes(num_latent + num_trash, reps=reps), range(0, num_latent + num_trash), inplace=True)
    qc.barrier()
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def domain_wall(circuit: QuantumCircuit, start: int, end: int) -> QuantumCircuit:
    """Apply X gates on a contiguous block to create a domain wall."""
    for qubit in range(start, end):
        circuit.x(qubit)
    return circuit

def fidelity_loss(pred: np.ndarray, target: np.ndarray) -> float:
    """Return 1 – fidelity between two state‑vectors."""
    return 1.0 - np.abs(np.vdot(target, pred))**2

def Autoencoder(
    num_latent: int = 3,
    num_trash: int = 2,
    reps: int = 5,
    seed: int = 42,
) -> SamplerQNN:
    """Factory returning a quantum auto‑encoder SamplerQNN."""
    _random_seed(seed)
    base_circuit = auto_encoder_circuit(num_latent, num_trash, reps)
    dw_circuit = domain_wall(QuantumCircuit(num_latent + 2 * num_trash + 1), 0, num_latent + 2 * num_trash + 1)
    full_circuit = dw_circuit.compose(base_circuit)
    sampler = StatevectorSampler()
    qnn = SamplerQNN(
        circuit=full_circuit,
        input_params=[],
        weight_params=full_circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )
    return qnn

def train_qautoencoder(
    qnn: SamplerQNN,
    target_state: np.ndarray,
    *,
    epochs: int = 50,
    learning_rate: float = 0.1,
    optimizer_cls=COBYLA,
    seed: int = 42,
) -> List[float]:
    """Train the quantum auto‑encoder to match a target state via fidelity loss."""
    _random_seed(seed)
    opt = optimizer_cls(maxiter=1000)
    loss_history: List[float] = []

    def objective(params: np.ndarray) -> float:
        qnn.weight_params = params
        probs = qnn.forward()
        amp = np.sqrt(probs[1]) if probs[1] >= 0 else 0.0
        state = np.array([np.sqrt(1 - amp**2), amp])
        loss = fidelity_loss(state, target_state)
        loss_history.append(loss)
        return loss

    opt.minimize(objective, qnn.weight_params)
    return loss_history

__all__ = ["Autoencoder", "train_qautoencoder"]
