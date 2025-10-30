import numpy as np
from dataclasses import dataclass
from typing import Iterable
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector, Pauli

@dataclass
class FraudLayerParameters:
    """Parameters for a single quantum fraud detection layer."""
    rzz_angle: float
    rx_angle: float
    ry_angle: float

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(circuit: QuantumCircuit, params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a parametric layer to the circuit."""
    rzz = params.rzz_angle
    rx = params.rx_angle
    ry = params.ry_angle
    if clip:
        rzz = _clip(rzz, 5.0)
        rx = _clip(rx, 5.0)
        ry = _clip(ry, 5.0)
    circuit.rzz(rzz, 0, 1)
    circuit.rx(rx, 0)
    circuit.ry(ry, 1)

def build_quantum_fraud_circuit(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    clip: bool = True
) -> QuantumCircuit:
    """Construct a parametric Qiskit circuit for fraud detection."""
    circuit = QuantumCircuit(2)
    # Parameterized input encoding
    inp1 = Parameter("inp1")
    inp2 = Parameter("inp2")
    circuit.ry(inp1, 0)
    circuit.ry(inp2, 1)
    # Input layer (unclipped)
    _apply_layer(circuit, input_params, clip=False)
    # Subsequent layers
    for idx, layer in enumerate(layers):
        rzz_p = Parameter(f"rzz_{idx}")
        rx_p = Parameter(f"rx_{idx}")
        ry_p = Parameter(f"ry_{idx}")
        circuit.rzz(rzz_p, 0, 1)
        circuit.rx(rx_p, 0)
        circuit.ry(ry_p, 1)
    return circuit

class QuantumFraudFeatureExtractor:
    """Evaluate a Qiskit circuit to produce two quantum features."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True
    ) -> None:
        self.circuit_template = build_quantum_fraud_circuit(
            input_params=input_params,
            layers=layers,
            clip=clip
        )
        self.simulator = Aer.get_backend("statevector_simulator")
        # Bind layer parameters to the template
        self.layer_bindings = {}
        for idx, layer in enumerate(layers):
            self.layer_bindings[f"rzz_{idx}"] = layer.rzz_angle
            self.layer_bindings[f"rx_{idx}"] = layer.rx_angle
            self.layer_bindings[f"ry_{idx}"] = layer.ry_angle

    def _evaluate_single(self, inp: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a single input pair."""
        bindings = {
            "inp1": inp[0],
            "inp2": inp[1],
            **self.layer_bindings
        }
        bound_circ = self.circuit_template.bind_parameters(bindings)
        result = execute(bound_circ, self.simulator, shots=1).result()
        state = Statevector(result.get_statevector(bound_circ))
        y_op = np.array([[0, -1j], [1j, 0]])
        exp_y0 = np.real(state.expectation_value(y_op, qubits=[0]))
        exp_y1 = np.real(state.expectation_value(y_op, qubits=[1]))
        return np.array([exp_y0, exp_y1], dtype=np.float32)

    def batch_evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of inputs."""
        return np.vstack([self._evaluate_single(inp) for inp in inputs])

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Convenience alias for batch_evaluate."""
        return self.batch_evaluate(inputs)

class FraudDetectorHybrid:
    """Quantum part of the fraud detection model.

    Provides a method to evaluate the parametric Qiskit circuit and
    return two quantum features for each input pair.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True
    ) -> None:
        self.extractor = QuantumFraudFeatureExtractor(
            input_params=input_params,
            layers=layers,
            clip=clip
        )

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """Return quantum features for a batch of inputs."""
        return self.extractor.evaluate(inputs)

    def build_circuit(self) -> QuantumCircuit:
        """Return the underlying Qiskit circuit template."""
        return self.extractor.circuit_template

# EstimatorQNN integration (optional)
def EstimatorQNN(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
):
    """Return a Qiskit EstimatorQNN object that uses the fraud circuit."""
    from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
    from qiskit.primitives import Estimator
    circuit = build_quantum_fraud_circuit(input_params, layers)
    # Observable: Pauli Y on both qubits
    observable = Pauli("YY")
    estimator = Estimator()
    return QiskitEstimatorQNN(
        circuit=circuit,
        observables=observable,
        input_params=[Parameter("inp1"), Parameter("inp2")],
        weight_params=[],
        estimator=estimator
    )

__all__ = [
    "FraudLayerParameters",
    "QuantumFraudFeatureExtractor",
    "FraudDetectorHybrid",
    "EstimatorQNN"
]
