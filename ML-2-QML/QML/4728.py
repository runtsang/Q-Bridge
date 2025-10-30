"""Hybrid quantum model mirroring the classical HybridConvFraudKernel implementation."""

from __future__ import annotations

import numpy as np
import qiskit
import strawberryfields as sf
from strawberryfields.ops import BSgate, Rgate, Sgate, Dgate, Kgate
import torchquantum as tq
from torchquantum.functional import func_name_dict


# ------------------------------------------------------------------  Quanvolution  ------------------------------------
class QuanvCircuit:
    """Quantum convolution filter using a randomized circuit."""
    def __init__(self, kernel_size: int = 2, backend=None, shots: int = 200, threshold: float = 127.0):
        self.n_qubits = kernel_size ** 2
        self._circuit = qiskit.QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._circuit.rx(self.theta[i], i)
        self._circuit.barrier()
        self._circuit += qiskit.circuit.random.random_circuit(self.n_qubits, 2)
        self._circuit.measure_all()

        self.backend = backend or qiskit.Aer.get_backend("qasm_simulator")
        self.shots = shots
        self.threshold = threshold

    def run(self, data: np.ndarray) -> float:
        data = np.reshape(data, (1, self.n_qubits))
        param_binds = []
        for dat in data:
            bind = {t: np.pi if v > self.threshold else 0 for t, v in zip(self.theta, dat)}
            param_binds.append(bind)

        job = qiskit.execute(self._circuit,
                             self.backend,
                             shots=self.shots,
                             parameter_binds=param_binds)
        result = job.result().get_counts(self._circuit)

        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val
        return counts / (self.shots * self.n_qubits)


# ------------------------------------------------------------------  Photonic Fraud Layer  ------------------------------------
class FraudLayerParameters:
    """Container matching the classical parameters for photonic layers."""
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> sf.Program:
    prog = sf.Program(2)
    with prog.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return prog


def _apply_layer(modes, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]


# ------------------------------------------------------------------  Quantum Kernel  ------------------------------------
class KernalAnsatz(tq.QuantumModule):
    """Encodes two classical vectors through a fixed list of single‑qubit rotations."""
    def __init__(self, func_list):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if tq.op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)


class Kernel(tq.QuantumModule):
    """Quantum kernel evaluated via the fixed ansatz."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])


# ------------------------------------------------------------------  Hybrid Class  ------------------------------------
class HybridConvFraudKernel:
    """
    Quantum analogue of the classical HybridConvFraudKernel.
    Provides methods that mimic the classical API.
    """
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 127.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: list[FraudLayerParameters] | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        self.conv = QuanvCircuit(kernel_size=conv_kernel_size,
                                 backend=qiskit.Aer.get_backend("qasm_simulator"),
                                 shots=200,
                                 threshold=conv_threshold)
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0, phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
        if fraud_layers is None:
            fraud_layers = []
        self.fraud_prog = build_fraud_detection_program(fraud_input_params, fraud_layers)
        self.kernel = Kernel()

    def forward(self, image: np.ndarray) -> float:
        """
        Run the quantum convolution, then the photonic fraud circuit.
        The photonic program is executed on a default Strawberry‑Fields simulator.
        """
        conv_out = self.conv.run(image)
        # Simulate the photonic circuit on a 2‑mode device
        sim = sf.backends.Simulator()
        result = sim.run(self.fraud_prog, args=sf.QubitRegister(2, 0))
        # Use the mean photon number of the first mode as a pseudo‑output
        mean_photon = result.samples.mean()
        return float(mean_photon)

    def compute_kernel(self, x: np.ndarray, y: np.ndarray) -> float:
        """Return the quantum kernel value between two vectors."""
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        return float(self.kernel(x_t, y_t).item())
