from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler, StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp

class HybridSamplerEstimatorQNN:
    """
    Hybrid quantum neural network that provides both sampling probabilities and expectation value.
    """

    def __init__(self) -> None:
        # Parameter vectors for sampler part
        self.inputs2 = ParameterVector("input_sampler", 2)
        self.weights2 = ParameterVector("weight_sampler", 4)

        # Parameter vectors for estimator part
        self.inputs1 = ParameterVector("input_estimator", 1)
        self.weights1 = ParameterVector("weight_estimator", 1)

        # Build circuit
        self.circuit = QuantumCircuit(2)

        # Sampler part on qubits 0 and 1
        self.circuit.ry(self.inputs2[0], 0)
        self.circuit.ry(self.inputs2[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights2[0], 0)
        self.circuit.ry(self.weights2[1], 1)
        self.circuit.cx(0, 1)
        self.circuit.ry(self.weights2[2], 0)
        self.circuit.ry(self.weights2[3], 1)

        # Estimator part on qubit 1
        self.circuit.h(1)
        self.circuit.ry(self.inputs1[0], 1)
        self.circuit.rx(self.weights1[0], 1)

        # Observable for estimator
        self.observable = SparsePauliOp.from_list([("Y", 1)])

        # Primitives
        self.sampler = StatevectorSampler()
        self.estimator = StatevectorEstimator()

        # Wrap into Qiskit neural network objects
        self.sampler_qnn = SamplerQNN(
            circuit=self.circuit,
            input_params=self.inputs2,
            weight_params=self.weights2,
            sampler=self.sampler
        )

        self.estimator_qnn = EstimatorQNN(
            circuit=self.circuit,
            observables=self.observable,
            input_params=self.inputs1,
            weight_params=self.weights1,
            estimator=self.estimator
        )

    def predict(self, input_values: dict) -> dict:
        """
        Evaluate the hybrid QNN.
        Parameters
        ----------
        input_values : dict
            Keys:
                'input_sampler' (list of 2 floats),
                'input_estimator' (list of 1 float),
                'weight_sampler' (list of 4 floats),
                'weight_estimator' (list of 1 float).
        Returns
        -------
        dict
            {'sampling': probabilities (np.ndarray), 'estimation': expectation value (float)}
        """
        # Bind parameters
        bound_circuit = self.circuit.bind_parameters({
            **{p: v for p, v in zip(self.inputs2, input_values["input_sampler"])},
            **{p: v for p, v in zip(self.weights2, input_values["weight_sampler"])},
            **{p: v for p, v in zip(self.inputs1, input_values["input_estimator"])},
            **{p: v for p, v in zip(self.weights1, input_values["weight_estimator"])},
        })

        # Sampling probabilities
        probs = self.sampler_qnn.predict(
            bound_circuit,
            input_values["input_sampler"],
            input_values["weight_sampler"]
        )

        # Estimation expectation value
        exp_val = self.estimator_qnn.predict(
            bound_circuit,
            input_values["input_estimator"],
            input_values["weight_estimator"]
        )

        return {"sampling": probs, "estimation": exp_val}

def SamplerQNN() -> HybridSamplerEstimatorQNN:
    """
    Compatibility wrapper returning an instance of HybridSamplerEstimatorQNN.
    """
    return HybridSamplerEstimatorQNN()

__all__ = ["HybridSamplerEstimatorQNN", "SamplerQNN"]
