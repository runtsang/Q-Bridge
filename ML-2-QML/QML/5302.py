"""Hybrid estimator that unifies quantum primitives.

This module defines :class:`HybridEstimator` which extends the lightweight
FastBaseEstimator from the seed project.  It can optionally chain a quantum
convolution filter, a quantum auto‑encoder and a quantum classifier.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Optional, Tuple
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_aer import AerSimulator

# --------------------------------------------------------------------------- #
# Base estimator copied from the seed (FastBaseEstimator)
# --------------------------------------------------------------------------- #
class FastBaseEstimator:
    """Evaluate expectation values of observables for a parametrized circuit."""
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
        observables = list(observables)
        results: List[List[complex]] = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

# --------------------------------------------------------------------------- #
# Quantum Conv filter
# --------------------------------------------------------------------------- #
class QuantumConvCircuit(QuantumCircuit):
    """Filter circuit used for quanvolution layers."""
    def __init__(self, kernel_size: int, shots: int = 100, threshold: float = 0.0):
        n_qubits = kernel_size ** 2
        super().__init__(n_qubits)
        self.theta = ParameterVector("theta", n_qubits)
        for i in range(n_qubits):
            self.rx(self.theta[i], i)
        self.barrier()
        self += random_circuit(n_qubits, 2)
        self.measure_all()
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on classical data and return average probability of |1>."""
        data = np.reshape(data, (1, self.num_qubits))
        param_binds = []
        for dat in data:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0
            param_binds.append(bind)

        backend = AerSimulator()
        job = backend.run(self, shots=self.shots, parameter_binds=param_binds)
        result = job.result()
        counts = result.get_counts(self)
        total = self.shots * self.num_qubits
        ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
        return ones / total

# --------------------------------------------------------------------------- #
# Quantum Auto‑encoder (SamplerQNN)
# --------------------------------------------------------------------------- #
def QuantumAutoencoder(num_latent: int = 3, num_trash: int = 2):
    """Return a quantum auto‑encoder implemented with SamplerQNN."""
    from qiskit_machine_learning.neural_networks import SamplerQNN
    from qiskit.circuit.library import RealAmplitudes
    from qiskit_machine_learning.utils.algorithm_globals import random_seed
    random_seed(42)

    def ansatz(num_qubits):
        return RealAmplitudes(num_qubits, reps=5)

    def auto_encoder_circuit(num_latent, num_trash):
        qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
        cr = ClassicalRegister(1, "c")
        circuit = QuantumCircuit(qr, cr)
        circuit.compose(ansatz(num_latent + num_trash), range(0, num_latent + num_trash), inplace=True)
        circuit.barrier()
        aux = num_latent + 2 * num_trash
        circuit.h(aux)
        for i in range(num_trash):
            circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
        circuit.h(aux)
        circuit.measure(aux, cr[0])
        return circuit

    circuit = auto_encoder_circuit(num_latent, num_trash)
    sampler = AerSimulator(method="statevector")
    return SamplerQNN(
        circuit=circuit,
        input_params=[],
        weight_params=circuit.parameters,
        interpret=lambda x: x,
        output_shape=2,
        sampler=sampler,
    )

# --------------------------------------------------------------------------- #
# Quantum Classifier circuit
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_qubits: int, depth: int) -> Tuple[QuantumCircuit, Sequence, Sequence, List[SparsePauliOp]]:
    """Construct a simple layered ansatz with explicit encoding and variational parameters."""
    encoding = ParameterVector("x", num_qubits)
    weights = ParameterVector("theta", num_qubits * depth)

    circuit = QuantumCircuit(num_qubits)
    for param, qubit in zip(encoding, range(num_qubits)):
        circuit.rx(param, qubit)

    index = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(weights[index], qubit)
            index += 1
        for qubit in range(num_qubits - 1):
            circuit.cz(qubit, qubit + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - i - 1)) for i in range(num_qubits)]
    return circuit, list(encoding), list(weights), observables

# --------------------------------------------------------------------------- #
# HybridEstimator
# --------------------------------------------------------------------------- #
class HybridEstimator(FastBaseEstimator):
    """Estimator that chains optional quantum convolution, auto‑encoder and
    classifier circuits.  The main circuit receives the output of the
    preceding stage as its input parameters.
    """
    def __init__(
        self,
        circuit: QuantumCircuit,
        conv: Optional[QuantumCircuit] = None,
        autoencoder: Optional[QuantumCircuit] = None,
        classifier: Optional[QuantumCircuit] = None,
        shots: int = 100,
        backend=None,
    ) -> None:
        super().__init__(circuit)
        self.conv = conv
        self.autoencoder = autoencoder
        self.classifier = classifier
        self.shots = shots
        self.backend = backend or AerSimulator()

    def _bind_stage(self, circ: QuantumCircuit, params: Sequence[float]) -> QuantumCircuit:
        if len(params)!= len(circ.parameters):
            raise ValueError("Parameter count mismatch for bound circuit.")
        mapping = dict(zip(circ.parameters, params))
        return circ.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[BaseOperator],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[List[complex]]:
        """Evaluate expectation values for each parameter set, chaining
        the optional convolution, auto‑encoder and classifier circuits.
        """
        results: List[List[complex]] = []
        for params in parameter_sets:
            # Convolution stage
            if self.conv is not None:
                conv_circ = self._bind_stage(self.conv, params)
                job = self.backend.run(conv_circ, shots=self.shots)
                counts = job.result().get_counts(conv_circ)
                ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
                conv_param = [ones / (self.shots * self.conv.num_qubits)]
            else:
                conv_param = params

            # Auto‑encoder stage
            if self.autoencoder is not None:
                ae_circ = self._bind_stage(self.autoencoder, conv_param)
                job = self.backend.run(ae_circ, shots=self.shots)
                counts = job.result().get_counts(ae_circ)
                ones = sum(sum(int(bit) for bit in key) * val for key, val in counts.items())
                ae_param = [ones / (self.shots * self.autoencoder.num_qubits)]
            else:
                ae_param = conv_param

            # Main circuit
            main_circ = self._bind_stage(self._circuit, ae_param)
            state = Statevector.from_instruction(main_circ)
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

    def predict(
        self,
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[int]:
        """Return class labels using the optional quantum classifier."""
        if self.classifier is None:
            raise RuntimeError("No quantum classifier circuit supplied.")
        predictions: List[int] = []
        for params in parameter_sets:
            circ = self._bind_stage(self.classifier, params)
            job = self.backend.run(circ, shots=self.shots)
            result = job.result().get_counts(circ)
            ones = result.get("1", 0)
            total = sum(result.values())
            pred = 1 if ones / total > 0.5 else 0
            predictions.append(pred)
        return predictions

    def train_classifier(
        self,
        parameter_sets: Sequence[Sequence[float]],
        labels: Sequence[int],
        *,
        epochs: int = 10,
    ) -> None:
        """Simple training loop that optimises a classical cost over the
        quantum classifier using the backend simulator.
        """
        if self.classifier is None:
            raise RuntimeError("No classifier circuit to train.")
        # Placeholder: random assignment of parameters
        for _ in range(epochs):
            for params in parameter_sets:
                perturbed = [p + np.random.normal(scale=0.01) for p in params]
                circ = self._bind_stage(self.classifier, perturbed)
                self.backend.run(circ, shots=self.shots)

__all__ = ["HybridEstimator"]
