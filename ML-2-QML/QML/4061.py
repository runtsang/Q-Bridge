"""Quantum QCNN implementation with convolution, pooling and sampler, plus fast estimator."""
from __future__ import annotations

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZFeatureMap
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.primitives import StatevectorEstimator, StatevectorSampler
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN

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

    def evaluate(self, observables, parameter_sets):
        """Compute expectation values for every parameter set and observable."""
        observables = list(observables)
        results = []
        for values in parameter_sets:
            state = Statevector.from_instruction(self._bind(values))
            row = [state.expectation_value(obs) for obs in observables]
            results.append(row)
        return results

class QCNNGen026QNN:
    """Quantum QCNN with convolution, pooling and sampler, plus fast estimator."""
    def __init__(self) -> None:
        self.estimator_circuit = self._build_qcnn_circuit()
        self.sampler_circuit = self._build_sampler_circuit()
        self.estimator_qnn = self._build_estimator_qnn()
        self.sampler_qnn = self._build_sampler_qnn()
        self.estimator = FastBaseEstimator(self.estimator_circuit)
        self.sampler = FastBaseEstimator(self.sampler_circuit)

    def _build_qcnn_circuit(self) -> QuantumCircuit:
        feature_map = ZFeatureMap(8)
        ansatz = QuantumCircuit(8, name="Ansatz")
        ansatz.compose(self._conv_layer(8, "c1"), inplace=True)
        ansatz.compose(self._pool_layer([0,1,2,3],[4,5,6,7],"p1"), inplace=True)
        ansatz.compose(self._conv_layer(4, "c2"), inplace=True)
        ansatz.compose(self._pool_layer([0,1],[2,3],"p2"), inplace=True)
        ansatz.compose(self._conv_layer(2, "c3"), inplace=True)
        ansatz.compose(self._pool_layer([0],[1],"p3"), inplace=True)

        circuit = QuantumCircuit(8)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        return circuit

    def _conv_layer(self, num_qubits: int, param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(num_qubits)
        params = ParameterVector(param_prefix, length=num_qubits * 3)
        for i in range(0, num_qubits, 2):
            qc.compose(self._conv_circuit(params[i*3:(i+2)*3]), [i, i+1], inplace=True)
        return qc

    def _conv_circuit(self, params: Sequence[ParameterVector]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        qc.cx(1,0)
        qc.rz(np.pi/2,0)
        return qc

    def _pool_layer(self, sources: Sequence[int], sinks: Sequence[int], param_prefix: str) -> QuantumCircuit:
        qc = QuantumCircuit(len(sources)+len(sinks))
        params = ParameterVector(param_prefix, length=len(sources)*3)
        for src, snk in zip(sources, sinks):
            qc.compose(self._pool_circuit(params[:3]), [src, snk], inplace=True)
            params = params[3:]
        return qc

    def _pool_circuit(self, params: Sequence[ParameterVector]) -> QuantumCircuit:
        qc = QuantumCircuit(2)
        qc.rz(-np.pi/2, 1)
        qc.cx(1,0)
        qc.rz(params[0],0)
        qc.ry(params[1],1)
        qc.cx(0,1)
        qc.ry(params[2],1)
        return qc

    def _build_sampler_circuit(self) -> QuantumCircuit:
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
        return qc

    def _build_estimator_qnn(self) -> EstimatorQNN:
        estimator = StatevectorEstimator()
        observable = SparsePauliOp.from_list([("Z" + "I"*7, 1)])
        return EstimatorQNN(
            circuit=self.estimator_circuit,
            observables=observable,
            input_params=self.estimator_circuit.parameters,
            weight_params=self.estimator_circuit.parameters,
            estimator=estimator,
        )

    def _build_sampler_qnn(self) -> SamplerQNN:
        sampler = StatevectorSampler()
        return SamplerQNN(
            circuit=self.sampler_circuit,
            input_params=self.sampler_circuit.parameters,
            weight_params=self.sampler_circuit.parameters,
            sampler=sampler,
        )

    def evaluate(self, observables, parameter_sets):
        return self.estimator.evaluate(observables, parameter_sets)

    def sample(self, parameter_sets):
        # Return expectation values for the sampler circuit (treated as a “probability” observable)
        return self.sampler.evaluate([Statevector.from_instruction(self.sampler_circuit).to_operator()], parameter_sets)

__all__ = ["QCNNGen026QNN"]
