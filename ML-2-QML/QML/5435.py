import qiskit
from qiskit.circuit import QuantumCircuit, ParameterVector
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.primitives import Estimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN as QSamplerQNN
from qiskit_machine_learning.optimizers import COBYLA
from typing import Iterable, List, Sequence, Union, Dict, Any

class FastBaseEstimator:
    """Unified estimator for classical and quantum primitives."""
    def __init__(self, backend: Union[QuantumCircuit, EstimatorQNN, QSamplerQNN]) -> None:
        self.backend = backend
        if isinstance(backend, QuantumCircuit):
            self._parameters = list(backend.parameters)
        else:
            self._parameters = None

    def _bind(self, params: Sequence[float]) -> QuantumCircuit:
        if not isinstance(self.backend, QuantumCircuit):
            raise TypeError("Parameter binding only supported for raw QuantumCircuit")
        if len(params)!= len(self._parameters):
            raise ValueError("Parameter count mismatch")
        mapping = dict(zip(self._parameters, params))
        return self.backend.assign_parameters(mapping, inplace=False)

    def evaluate(
        self,
        observables: Iterable[Any],
        parameter_sets: Sequence[Sequence[float]],
    ) -> List[Any]:
        """Dispatch evaluation to the underlying backend."""
        results: List[Any] = []
        if isinstance(self.backend, QuantumCircuit):
            for params in parameter_sets:
                state = Statevector.from_instruction(self._bind(params))
                row = [state.expectation_value(obs) for obs in observables]
                results.append(row)
        elif isinstance(self.backend, EstimatorQNN):
            param_dicts = [{p: v for p, v in zip(self.backend.weight_params, params)} for params in parameter_sets]
            eval_res = self.backend.evaluate(param_dicts)
            results.extend(eval_res)
        elif isinstance(self.backend, QSamplerQNN):
            sampler = StatevectorSampler()
            for params in parameter_sets:
                param_dict = {p: v for p, v in zip(self.backend.input_params, params)}
                sample = sampler.sample(self.backend.circuit, param_dict, shots=1000)
                probs = {out[0]: out[1] for out in sample}
                results.append(probs)
        else:
            raise TypeError("Unsupported backend type")
        return results

def QCNN() -> EstimatorQNN:
    """Quantum convolution‑based neural network (QCNN) from Qiskit."""
    from qiskit.circuit.library import ZFeatureMap
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit.primitives import Estimator

    feature_map = ZFeatureMap(8)
    ansatz = QuantumCircuit(8, name="Ansatz")
    # Convolution and pooling layers are omitted for brevity
    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)
    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=circuit.decompose(),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        estimator=Estimator(),
    )
    return qnn

def SamplerQNN() -> QSamplerQNN:
    """Quantum sampler neural network."""
    inputs = ParameterVector("input", 2)
    weights = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inputs[0], 0)
    qc.ry(inputs[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[0], 0)
    qc.ry(weights[1], 1)
    qc.cx(0, 1)
    qc.ry(weights[2], 0)
    qc.ry(weights[3], 1)
    sampler_qnn = QSamplerQNN(
        circuit=qc,
        input_params=inputs,
        weight_params=weights,
    )
    return sampler_qnn

__all__ = [
    "FastBaseEstimator",
    "QCNN",
    "SamplerQNN",
]
