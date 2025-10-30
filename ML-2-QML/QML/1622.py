"""Enhanced quantum neural network.

This module builds on the original EstimatorQNN example by:
  • Using a 2‑qubit variational circuit with entanglement.
  • Parameterising both input and weight angles.
  • Measuring a Pauli‑Z observable on the first qubit.
  • Wrapping the Qiskit EstimatorQNN for seamless integration.
"""

from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import StatevectorEstimator
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN

class EnhancedEstimatorQNN:
    """
    A quantum feed‑forward estimator that mimics the API of the classical
    EnhancedEstimatorQNN. It exposes a ``forward`` method that returns the
    expectation value of a Pauli‑Z observable on a 2‑qubit variational circuit.
    """

    def __init__(self, estimator: StatevectorEstimator | None = None) -> None:
        # ----- Build the parameterised circuit -----
        self.input_params = [Parameter("x1"), Parameter("x2")]
        self.weight_params = [Parameter("w1"), Parameter("w2"), Parameter("w3"), Parameter("w4")]

        qc = QuantumCircuit(2)
        # Input encoding
        qc.ry(self.input_params[0], 0)
        qc.ry(self.input_params[1], 1)

        # Variational layer 1
        qc.ry(self.weight_params[0], 0)
        qc.ry(self.weight_params[1], 1)
        qc.cx(0, 1)

        # Variational layer 2
        qc.ry(self.weight_params[2], 0)
        qc.ry(self.weight_params[3], 1)
        qc.cx(1, 0)

        # Observable: Z on qubit 0
        observable = SparsePauliOp.from_list([("Z" * qc.num_qubits, 1)])

        # Wrap with Qiskit EstimatorQNN
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=qc,
            observables=observable,
            input_params=self.input_params,
            weight_params=self.weight_params,
            estimator=estimator or StatevectorEstimator()
        )

    def forward(self, inputs: dict[str, float]) -> float:
        """
        Evaluate the quantum circuit for the given input angles.

        Parameters
        ----------
        inputs : dict
            Mapping from the names of the input parameters (``x1`` and ``x2``) to
            their numeric values.

        Returns
        -------
        float
            Expectation value of the observable.
        """
        return float(self.estimator_qnn.predict(inputs))

def EstimatorQNN() -> EnhancedEstimatorQNN:
    """
    Public factory that mirrors the classical EstimatorQNN signature.
    Returns an instance ready for use in a hybrid training loop.
    """
    return EnhancedEstimatorQNN()

__all__ = ["EnhancedEstimatorQNN", "EstimatorQNN"]
