import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit import Parameter
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

class Conv:
    """
    Quantum convolutional filter that encodes a 2‑D patch onto a qubit grid,
    applies a random circuit, and a photonic‑inspired variational layer.
    The output is the average probability of measuring |1> across all qubits.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        backend=None,
        shots: int = 100,
        threshold: float = 127,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend or Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

        # Parameterised data encoding
        self.theta = [Parameter(f"theta{i}") for i in range(self.n_qubits)]
        self._circuit = QuantumCircuit(self.n_qubits)

        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)

        self._circuit.barrier()

        # Random circuit to introduce entanglement
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)

        # Variational layer inspired by the photonic fraud‑detection circuit
        if fraud_params is None:
            fraud_params = []

        for params in fraud_params:
            self._apply_fraud_layer(params)

        self._circuit.measure_all()

    def _apply_fraud_layer(self, params: FraudLayerParameters) -> None:
        """Append a set of parameter‑clipped rotations to each qubit."""
        for i in range(self.n_qubits):
            bs_theta = _clip(params.bs_theta, 5.0)
            bs_phi = _clip(params.bs_phi, 5.0)
            phase0 = _clip(params.phases[0], 5.0)
            phase1 = _clip(params.phases[1], 5.0)
            squeeze_r0 = _clip(params.squeeze_r[0], 5.0)
            squeeze_r1 = _clip(params.squeeze_r[1], 5.0)
            squeeze_phi0 = _clip(params.squeeze_phi[0], 5.0)
            squeeze_phi1 = _clip(params.squeeze_phi[1], 5.0)
            disp_r0 = _clip(params.displacement_r[0], 5.0)
            disp_r1 = _clip(params.displacement_r[1], 5.0)
            disp_phi0 = _clip(params.displacement_phi[0], 5.0)
            disp_phi1 = _clip(params.displacement_phi[1], 5.0)
            kerr0 = _clip(params.kerr[0], 1.0)
            kerr1 = _clip(params.kerr[1], 1.0)

            self._circuit.rx(bs_theta, i)
            self._circuit.rz(bs_phi, i)
            if i == 0:
                self._circuit.rz(phase0, i)
                self._circuit.rz(squeeze_r0, i)
                self._circuit.rz(squeeze_phi0, i)
                self._circuit.rz(disp_r0, i)
                self._circuit.rz(disp_phi0, i)
                self._circuit.rz(kerr0, i)
            else:
                self._circuit.rz(phase1, i)
                self._circuit.rz(squeeze_r1, i)
                self._circuit.rz(squeeze_phi1, i)
                self._circuit.rz(disp_r1, i)
                self._circuit.rz(disp_phi1, i)
                self._circuit.rz(kerr1, i)

    def run(self, data: np.ndarray) -> float:
        """
        Execute the circuit on a single 2‑D patch.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (kernel_size, kernel_size) with integer pixel values.

        Returns
        -------
        float
            Average probability of measuring |1> across all qubits.
        """
        data_flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data_flat:
            bind = {self.theta[i]: np.pi if val > self.threshold else 0.0 for i, val in enumerate(dat)}
            param_binds.append(bind)

        job = execute(
            self._circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=param_binds,
        )
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (self.shots * self.n_qubits)

__all__ = ["Conv", "FraudLayerParameters"]
