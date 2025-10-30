import numpy as np
import strawberryfields as sf
from strawberryfields import ops as sfops
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    sfops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sfops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        sfops.Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    sfops.BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        sfops.Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        sfops.Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        sfops.Kgate(k if not clip else _clip(k, 1)) | modes[i]

class FraudDetectionHybrid:
    """
    Quantum sub‑model that combines a photonic circuit (using
    Strawberry Fields) with a Qiskit EstimatorQNN.  It exposes an
    ``evaluate`` method that accepts a batch of 2‑dimensional inputs
    and returns a 3‑dimensional feature vector per sample:
    [photonic X‑quadrature mean (mode 0), photonic X‑quadrature mean (mode 1),
     Y‑observable mean].
    """
    def __init__(self, photonic_params: Iterable[FraudLayerParameters]) -> None:
        # Build the static photonic program
        params_list = list(photonic_params)
        self.photonic_prog = sf.Program(2)
        with self.photonic_prog.context as q:
            _apply_layer(q, params_list[0], clip=False)
            for layer in params_list[1:]:
                _apply_layer(q, layer, clip=True)

        # Prepare Qiskit EstimatorQNN circuit
        input_param = Parameter("x")
        weight_param = Parameter("w")
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.ry(input_param, 0)
        qc.rx(weight_param, 0)
        self.qc = qc
        self.input_param = input_param
        self.weight_param = weight_param
        # Observable Y
        self.observable = SparsePauliOp.from_list([("Y" * qc.num_qubits, 1)])
        # Statevector estimator
        self.estimator = StatevectorEstimator()

    def _photonic_expectation(self) -> np.ndarray:
        """Return mean X quadrature for each mode (averaged)."""
        sim = sf.Simulator("gaussian")
        results = sim.run(self.photonic_prog)
        mean = results.state.mean_vector  # shape (4,): [x1, p1, x2, p2]
        return np.array([mean[0], mean[2]])

    def evaluate(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the 3‑dimensional feature vector for each input sample.
        Parameters:
            inputs: array shape (batch, 2) – first column is the
                    input rotation angle, second column is the weight
                    to be trained.
        Returns:
            features: array shape (batch, 3)
        """
        photonic_feat = self._photonic_expectation()
        batch_size = inputs.shape[0]
        y_expectations = np.empty(batch_size)
        for i, (x_val, w_val) in enumerate(inputs):
            bound_qc = self.qc.bind_parameters({
                self.input_param: x_val,
                self.weight_param: w_val
            })
            y_expectations[i] = self.estimator.run(bound_qc, self.observable).values[0]
        photonic_batch = np.tile(photonic_feat, (batch_size, 1))
        return np.concatenate([photonic_batch, y_expectations[:, None]], axis=1)
