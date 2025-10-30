import numpy as np
import qiskit
from qiskit.circuit.random import random_circuit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate
from dataclasses import dataclass
from typing import Sequence

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

class FraudDetectionHybrid:
    """Quantum‑augmented fraud‑detection model that fuses a qubit‑based
    quanvolution pre‑processor with a photonic variational circuit."""
    def __init__(self, shots: int = 100, backend=None):
        self.shots = shots
        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.threshold = 127

    def _build_quanv_circuit(self, n_qubits: int):
        circ = qiskit.QuantumCircuit(n_qubits)
        theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(n_qubits)]
        for i in range(n_qubits):
            circ.rx(theta[i], i)
        circ.barrier()
        circ += random_circuit(n_qubits, 2)
        circ.measure_all()
        return circ, theta

    def _run_quanv(self, data: np.ndarray, threshold: float) -> float:
        kernel_size = data.shape[0]
        circ, theta = self._build_quanv_circuit(kernel_size ** 2)
        param_binds = []
        for val in data.flatten():
            bind = {theta[i]: np.pi if val > threshold else 0}
            param_binds.append(bind)
        job = qiskit.execute(circ, self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(circ)
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * kernel_size ** 2)

    def _clip(self, value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_layer(self, modes: Sequence, params, *, clip: bool):
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

    def run(self, data: np.ndarray, input_params, layers) -> float:
        """Execute the hybrid circuit on a single 2‑D fraud‑sample."""
        # Two quanvolution features with slightly different thresholds
        f1 = self._run_quanv(data, self.threshold)
        f2 = self._run_quanv(data, self.threshold + 20)

        prog = sf.Program(2)
        with prog.context as q:
            # encode the two classical features as displacements
            Dgate(f1) | q[0]
            Dgate(f2) | q[1]
            self._apply_layer(q, input_params, clip=False)
            for layer in layers:
                self._apply_layer(q, layer, clip=True)
            # final measurement by a simple displacement
            Dgate(0) | q[0]
            Dgate(0) | q[1]
        eng = sf.Engine("fock", backend_options={"cutoff_dim": 10})
        result = eng.run(prog)
        # use photon number expectation of first mode as output
        return result.samples[0, 0].mean()

__all__ = ["FraudDetectionHybrid"]
