from dataclasses import dataclass
from typing import Iterable, Sequence
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit.primitives import StatevectorSampler as Sampler, StatevectorEstimator as Estimator
from qiskit.quantum_info import SparsePauliOp

@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(val: float, bound: float) -> float:
    """Clips a value to the interval [-bound, bound]."""
    return max(-bound, min(bound, val))

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Construct a Strawberry Fields program for fraud detection."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for lay in layers:
            _apply_layer(q, lay, clip=True)
    return prog

def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """Quantum auto‑encoder with a domain‑wall swap‑test."""
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)

    # Ansatz – simple H layers
    for i in range(num_latent + num_trash):
        qc.h(i)

    # Swap test
    aux = num_latent + 2 * num_trash
    qc.h(aux)
    for i in range(num_trash):
        qc.cswap(aux, num_latent + i, num_latent + num_trash + i)
    qc.h(aux)
    qc.measure(aux, cr[0])
    return qc

def sampler_qnn_circuit() -> QuantumCircuit:
    """Parameterised sampler circuit used in the hybrid pipeline."""
    inp = ParameterVector("input", 2)
    w = ParameterVector("weight", 4)
    qc = QuantumCircuit(2)
    qc.ry(inp[0], 0)
    qc.ry(inp[1], 1)
    qc.cx(0, 1)
    qc.ry(w[0], 0)
    qc.ry(w[1], 1)
    qc.cx(0, 1)
    qc.ry(w[2], 0)
    qc.ry(w[3], 1)
    return qc

def estimator_qnn_circuit() -> QuantumCircuit:
    """Parameterised estimator circuit for regression."""
    inp = Parameter("input")
    w = Parameter("weight")
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.ry(inp, 0)
    qc.rx(w, 0)
    return qc

class FraudDetectionHybridQuantum:
    """Hybrid quantum fraud detection pipeline."""
    def __init__(self, fraud_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        # Photonic fraud‑detection program
        self.fraud_prog = build_fraud_detection_program(fraud_params, layers)

        # Quantum auto‑encoder
        self.auto_circ = autoencoder_circuit(num_latent=3, num_trash=2)

        # Sampler and estimator QNNs
        self.sampler_circ = sampler_qnn_circuit()
        self.estimator_circ = estimator_qnn_circuit()

        # Qiskit primitives
        self.sampler = Sampler()
        self.estimator = Estimator()

        # Construct QNN wrappers
        self.sampler_qnn = SamplerQNN(
            circuit=self.sampler_circ,
            input_params=[Parameter("input0"), Parameter("input1")],
            weight_params=[Parameter("w0"), Parameter("w1"), Parameter("w2"), Parameter("w3")],
            sampler=self.sampler,
        )
        self.estimator_qnn = EstimatorQNN(
            circuit=self.estimator_circ,
            observables=SparsePauliOp.from_list([("Y", 1)]),
            input_params=[Parameter("input")],
            weight_params=[Parameter("weight")],
            estimator=self.estimator,
        )

    def evaluate(self, input_vec: list[float]) -> float:
        """
        Run the full quantum pipeline:
        1. Simulate the photonic fraud‑detection program.
        2. Feed the two‑mode output to the sampler QNN.
        3. Feed the sampler output to the estimator QNN.
        Returns the final fraud score.
        """
        # 1. Photonic simulation (placeholder – uses Strawberry Fields engine)
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(self.fraud_prog, data={"phi": [0.0, 0.0]})
        photonic_out = result.samples[0][:2]  # take first two modes

        # 2. Sampler QNN forward
        sampler_out = self.sampler_qnn.predict(photonic_out)

        # 3. Estimator QNN forward
        score = self.estimator_qnn.predict(sampler_out)
        return float(score[0][0])

__all__ = ["FraudDetectionHybridQuantum", "FraudLayerParameters"]
