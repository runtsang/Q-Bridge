import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit.providers.aer import AerSimulator
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit.primitives import StatevectorEstimator as Estimator
from sklearn.kernel_ridge import KernelRidge

class EstimatorQNN__gen484:
    """
    Quantum‑kernel ridge regressor that builds on the EstimatorQNN example.
    The circuit encodes two input features and a single trainable weight.
    The observable is a Pauli‑Y on the single qubit, and the kernel is evaluated
    using the StatevectorEstimator backend.
    """
    def __init__(self, lambda_reg: float = 1e-3) -> None:
        # Build the parameterised circuit
        self.input_params = [Parameter("x1"), Parameter("x2")]
        self.weight_param = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(self.input_params[0], 0)
        qc.rx(self.weight_param, 0)
        qc.ry(self.input_params[1], 0)
        # Observable
        observable = SparsePauliOp.from_list([("Y", 1)])
        # Backend and estimator
        backend = AerSimulator(method="statevector")
        estimator = Estimator(backend=backend)
        # Quantum kernel
        self.kernel = QuantumKernel(
            circuit=qc,
            observable=observable,
            estimator=estimator,
            backend=backend,
        )
        # Classical ridge regressor
        self.model = KernelRidge(alpha=lambda_reg, kernel="precomputed")
        self.X_train: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the quantum‑kernel ridge regressor.
        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, 2).
        y : np.ndarray
            Target values of shape (n_samples,).
        """
        K = self.kernel.evaluate(X, X)
        self.model.fit(K, y)
        self.X_train = X.copy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for new samples.
        Parameters
        ----------
        X : np.ndarray
            Test features of shape (m_samples, 2).
        Returns
        -------
        np.ndarray
            Predicted values of shape (m_samples,).
        """
        if self.X_train is None:
            raise RuntimeError("Model has not been fitted yet.")
        K_test = self.kernel.evaluate(X, self.X_train)
        return self.model.predict(K_test)

__all__ = ["EstimatorQNN__gen484"]
