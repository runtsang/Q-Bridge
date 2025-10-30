import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import Sampler
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.utils import algorithm_globals

# ------------------------------------------------------------------
# Quantum autoencoder circuit builder
# ------------------------------------------------------------------
def _auto_encoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Construct a swap‑test based quantum autoencoder."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    # Ansatz
    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)

    # Swap test
    auxiliary = num_latent + 2 * num_trash
    circuit.h(auxiliary)
    for i in range(num_trash):
        circuit.cswap(auxiliary, num_latent + i, num_latent + num_trash + i)
    circuit.h(auxiliary)
    circuit.measure(auxiliary, cr[0])
    return circuit

def _domain_wall_circuit(num_qubits: int, start: int, end: int) -> QuantumCircuit:
    """Flip qubits in the range [start, end)."""
    qc = QuantumCircuit(num_qubits)
    for i in range(start, end):
        qc.x(i)
    return qc

# ------------------------------------------------------------------
# Fast estimator for quantum circuits
# ------------------------------------------------------------------
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrised circuit."""
    def __init__(self, circuit: QuantumCircuit):
        self._circuit = circuit
        self._parameters = list(circuit.parameters)

    def _bind(self, parameter_values: Sequence[float]) -> QuantumCircuit:
        if len(parameter_values)!= len(self._parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(self._parameters, parameter_values))
        return self._circuit.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# ------------------------------------------------------------------
# Hybrid quantum autoencoder class
# ------------------------------------------------------------------
class AutoencoderGen317:
    """
    Quantum autoencoder exposing a FastEstimator‑style API.
    The class builds a swap‑test circuit, optionally augments it with a
    domain‑wall pattern, and wraps a SamplerQNN for fast forward passes.
    """
    def __init__(
        self,
        num_latent: int = 3,
        num_trash: int = 2,
        add_domain_wall: bool = True,
    ) -> None:
        self.num_latent = num_latent
        self.num_trash = num_trash
        self.circuit = _auto_encoder_circuit(num_latent, num_trash)

        if add_domain_wall:
            dw = _domain_wall_circuit(self.circuit.num_qubits, 0, self.circuit.num_qubits)
            self.circuit.compose(dw, inplace=True)

        # SamplerQNN for fast forward evaluation
        sampler = Sampler()
        self.qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=[],
            weight_params=self.circuit.parameters,
            interpret=lambda x: x,  # identity
            output_shape=(2,),
            sampler=sampler,
        )

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Delegate to the underlying FastBaseEstimator."""
        estimator = FastBaseEstimator(self.circuit)
        return estimator.evaluate(observables, parameter_sets)

    # ------------------------------------------------------------------
    # Convenience forward method (returns probability amplitudes)
    # ------------------------------------------------------------------
    def forward(self, parameter_values: Sequence[float]) -> np.ndarray:
        """Return the SamplerQNN output for a single parameter set."""
        return self.qnn.forward(np.array([parameter_values]))[0]

# ------------------------------------------------------------------
# Factory helper
# ------------------------------------------------------------------
def AutoencoderGen317_factory(
    num_latent: int = 3,
    num_trash: int = 2,
    add_domain_wall: bool = True,
) -> AutoencoderGen317:
    """Convenience factory mirroring the original QML `Autoencoder`."""
    return AutoencoderGen317(num_latent, num_trash, add_domain_wall)

__all__ = [
    "AutoencoderGen317",
    "AutoencoderGen317_factory",
    "FastBaseEstimator",
]
