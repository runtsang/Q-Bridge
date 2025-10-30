"""Hybrid quantum auto‑encoder + photonic fraud‑detection circuit.

The module exposes two variational circuits: a Qiskit Real‑Amplitudes auto‑encoder
and a Strawberry‑Fields photonic fraud detector.  A `HybridQuantumAutoFraudQNN`
combines them into a single SamplerQNN that can be used in a variational
algorithm.  The quantum part refines the latent representation while the
classical interpreter evaluates the fraud score using the photonic model.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorSampler as Sampler
from qiskit_machine_learning.neural_networks import SamplerQNN
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate


# ------------------------------------------------------------------ #
# Quantum auto‑encoder circuit (Qiskit)
# ------------------------------------------------------------------ #
def autoencoder_circuit(num_latent: int, num_trash: int) -> QuantumCircuit:
    """
    Variational auto‑encoder that maps an input state to a latent space
    and back using a Real‑Amplitudes ansatz.

    Parameters
    ----------
    num_latent : int
        Number of latent qubits.
    num_trash : int
        Number of auxiliary (trash) qubits.
    """
    qr = QuantumRegister(num_latent + 2 * num_trash + 1, "q")
    cr = ClassicalRegister(1, "c")
    circuit = QuantumCircuit(qr, cr)

    ansatz = RealAmplitudes(num_latent + num_trash, reps=5)
    circuit.compose(ansatz, range(0, num_latent + num_trash), inplace=True)
    circuit.barrier()

    # Swap‑test style entanglement with the auxiliary qubit
    aux = num_latent + 2 * num_trash
    circuit.h(aux)
    for i in range(num_trash):
        circuit.cswap(aux, num_latent + i, num_latent + num_trash + i)
    circuit.h(aux)
    circuit.measure(aux, cr[0])

    return circuit


# ------------------------------------------------------------------ #
# Photonic fraud‑detection program (Strawberry‑Fields)
# ------------------------------------------------------------------ #
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


def _apply_photonic_layer(
    modes: list, params: FraudLayerParameters, *, clip: bool
) -> None:
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


def photonic_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> sf.Program:
    """Return a Strawberry‑Fields program that implements the fraud detector."""
    prog = sf.Program(2)
    with prog.context as q:
        _apply_photonic_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_photonic_layer(q, layer, clip=True)
    return prog


# ------------------------------------------------------------------ #
# Hybrid QNN
# ------------------------------------------------------------------ #
class HybridQuantumAutoFraudQNN(SamplerQNN):
    """
    Combines the quantum auto‑encoder with the photonic fraud detector.
    The sampler returns the probability distribution over the auxiliary qubit,
    which is interpreted as a fraud score after passing through the photonic
    program (state‑vector simulation).
    """

    def __init__(
        self,
        num_latent: int,
        num_trash: int,
        fraud_params: Iterable[FraudLayerParameters],
        *,
        sampler: Sampler | None = None,
    ) -> None:
        circuit = autoencoder_circuit(num_latent, num_trash)
        super().__init__(
            circuit=circuit,
            input_params=[],
            weight_params=circuit.parameters,
            interpret=self._interpret,
            output_shape=2,
            sampler=sampler or Sampler(),
        )
        self.fraud_prog = photonic_fraud_detection_program(
            next(iter(fraud_params), FraudLayerParameters(0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0))),
            fraud_params,
        )

    def _interpret(self, sample: np.ndarray) -> float:
        """
        Interpret the sample from the quantum auto‑encoder as a fraud score.
        The sample is the probability of measuring |1⟩ on the auxiliary qubit.
        We feed the corresponding statevector to the photonic program and
        read out the mean photon number of mode 0 as a fraud indicator.
        """
        prob_one = sample[1]
        # Prepare the statevector with the given probability on |1⟩
        state = np.array([np.sqrt(1 - prob_one), np.sqrt(prob_one)], dtype=complex)
        eng = sf.Engine("gaussian")
        eng.run(self.fraud_prog, args={"phi": 0.0})
        mean_photon = eng.output_state.expectation_value(sf.ops.GaussianMode(0))
        # Map mean photon number to a fraud likelihood
        return float(mean_photon * prob_one)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Run the hybrid QNN on the provided inputs."""
        return super().forward(inputs)


__all__ = [
    "autoencoder_circuit",
    "photonic_fraud_detection_program",
    "FraudLayerParameters",
    "HybridQuantumAutoFraudQNN",
]
