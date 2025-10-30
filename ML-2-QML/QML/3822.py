from dataclasses import dataclass
from typing import Iterable, Sequence
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
import qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN as QiskitSamplerQNN
from qiskit.primitives import StatevectorSampler

@dataclass
class FraudLayerParameters:
    """Fully‑connected layer parameters adapted for a photonic circuit."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class HybridFraudDetection:
    """Quantum side of the hybrid fraud‑detection pipeline."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters]) -> None:
        self.sf_program = self._build_photonic(input_params, layers)
        self.qiskit_sampler = self._build_qubit(layers)

    def _build_photonic(self, input_params, layers):
        prog = sf.Program(2)
        with prog.context as q:
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)
        return prog

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            Sgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
        for i, phase in enumerate(params.phases):
            Rgate(phase) | modes[i]
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            Dgate(r if not clip else self._clip(r, 5), phi) | modes[i]
        for i, k in enumerate(params.kerr):
            Kgate(k if not clip else self._clip(k, 1)) | modes[i]

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _build_qubit(self, layers):
        n = 2
        qc = QuantumCircuit(n)
        input_params = ParameterVector('x', n)
        for i in range(n):
            qc.ry(input_params[i], i)
        qc.cx(0, 1)
        weight_params = ParameterVector('w', 4)
        for i in range(2):
            qc.ry(weight_params[i], i)
        qc.cx(0, 1)
        for i in range(2, 4):
            qc.ry(weight_params[i], i-2)
        sampler = StatevectorSampler()
        return QiskitSamplerQNN(circuit=qc,
                                input_params=input_params,
                                weight_params=weight_params,
                                sampler=sampler)

    def run(self, eng: sf.Engine = None) -> sf.Result:
        """Execute the photonic component; the qubit sampler is queried via its interface."""
        if eng is None:
            eng = sf.Engine('gaussian')
        return eng.run(self.sf_program)
