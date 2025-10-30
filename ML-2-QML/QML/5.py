from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN as QiskitEstimatorQNN
from qiskit.primitives import Estimator as StatevectorEstimator
import numpy as np

class EstimatorQNN:
    """
    Quantum regression model based on a parameterised ansatz and a
    multi‑qubit observable set.  The circuit depth, number of qubits and
    backend can be customised to trade off expressiveness against
    simulation cost.
    """

    def __init__(self,
                 num_qubits: int = 4,
                 depth: int = 3,
                 backend=None) -> None:
        self.num_qubits = num_qubits
        self.depth = depth
        self.circuit = self._build_ansatz()
        self.observables = self._build_observables()
        self.estimator = StatevectorEstimator(backend=backend)
        self.estimator_qnn = QiskitEstimatorQNN(
            circuit=self.circuit,
            observables=self.observables,
            input_params=[self.circuit.parameters[0]],
            weight_params=self.circuit.parameters[1:],
            estimator=self.estimator
        )

    def _build_ansatz(self) -> QuantumCircuit:
        """Construct a layered rotation‑plus‑entanglement ansatz."""
        qc = QuantumCircuit(self.num_qubits)
        params = []
        for layer in range(self.depth):
            for q in range(self.num_qubits):
                p_r = Parameter(f"theta_{layer}_{q}_r")
                p_z = Parameter(f"theta_{layer}_{q}_z")
                params.extend([p_r, p_z])
                qc.ry(p_r, q)
                qc.rz(p_z, q)
            # Entangling layer (full‑chain + wrap‑around)
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)
            qc.cx(self.num_qubits - 1, 0)
        return qc

    def _build_observables(self) -> list[SparsePauliOp]:
        """Return a list of single‑qubit Z observables."""
        return [SparsePauliOp.from_list([(f"Z{q}", 1)]) for q in range(self.num_qubits)]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """
        Predict regression outputs for an array of scalar inputs.

        Parameters
        ----------
        inputs : np.ndarray, shape (N, 1)
            Each sample is a single float that is injected into the first
            circuit parameter.
        """
        preds = []
        for x in inputs:
            param_dict = {self.circuit.parameters[0]: float(x[0])}
            val = self.estimator_qnn.predict(param_dict)
            preds.append(val[0])
        return np.array(preds)
__all__ = ["EstimatorQNN"]
