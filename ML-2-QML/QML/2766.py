import numpy as np
import qiskit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate, number
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

class QuantumConvFilter:
    """Quantum convolutional filter implemented with Qiskit."""
    def __init__(self, kernel_size: int = 2, shots: int = 200, threshold: float = 127.0):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()
        self.backend = qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        """Return the average probability of measuring |1> across qubits."""
        flat = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in flat:
            bind = {}
            for i, val in enumerate(dat):
                bind[self.theta[i]] = np.pi if val > self.threshold else 0.0
            param_binds.append(bind)
        job = qiskit.execute(
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

class HybridFraudDetector:
    """Hybrid quantum fraud‑detection pipeline that stitches a Qiskit convolution
    into a Strawberry Fields photonic circuit."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        conv_kernel: int = 2,
        conv_shots: int = 200,
        conv_threshold: float = 127.0,
    ) -> None:
        self.conv_filter = QuantumConvFilter(kernel_size=conv_kernel, shots=conv_shots, threshold=conv_threshold)
        self.input_params = input_params
        self.layers = list(layers)
        self.engine = sf.Engine("fock", backend_options={"cutoff_dim": 4})

    def _apply_layer(self, modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
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

    def run(self, data: np.ndarray) -> float:
        """Execute the full hybrid circuit on a single data patch."""
        # First obtain a quantum‑convolution value
        conv_val = self.conv_filter.run(data)

        # Build photonic program
        prog = sf.Program(2)
        with prog.context as q:
            # Encode conv_val as a displacement on both modes
            Dgate(conv_val, 0) | q[0]
            Dgate(conv_val, 0) | q[1]
            # Input photonic layer
            self._apply_layer(q, self.input_params, clip=False)
            # Subsequent layers
            for layer in self.layers:
                self._apply_layer(q, layer, clip=True)

        # Run simulation
        result = self.engine.run(prog)
        state = result.state
        # Expectation of photon number in mode 0
        exp_val = state.expectation(number(0))
        return float(exp_val)

__all__ = ["FraudLayerParameters", "HybridFraudDetector"]
