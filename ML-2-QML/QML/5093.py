import numpy as np
import torch
import qiskit
import torchquantum as tq
import strawberryfields as sf
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from torchquantum.functional import func_name_dict
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

class HybridQuantumFCL:
    """
    Quantum counterpart of HybridFCL.
    Implements:
        • Parameterised Qiskit circuit for a fully‑connected layer.
        • Qiskit self‑attention style block.
        • TorchQuantum RBF‑kernel ansatz.
        • StrawberryFields photonic program for fraud‑detection layers.
    All circuits can be executed on simulators and their expectation values
    are returned as NumPy arrays.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 backend_qiskit=None,
                 backend_sf: str = "fock",
                 sf_dim: int = 10):
        # Qiskit backend
        self.backend_qiskit = backend_qiskit or qiskit.Aer.get_backend("qasm_simulator")
        self.n_qubits = n_qubits

        # Fully‑connected Qiskit circuit
        self.fcl_circ = self._build_fcl_circuit()

        # Self‑attention Qiskit circuit
        self.attn_circ = self._build_attention_circuit()

        # TorchQuantum kernel
        self.kernel = self._build_kernel()

        # StrawberryFields program
        self.sf_dim = sf_dim
        self.backend_sf = sf.Engine(backend_sf, backend_args={"cutoff_dim": sf_dim})

    # ------------------------------------------------------------------
    # 1. Fully‑connected Qiskit circuit
    # ------------------------------------------------------------------
    def _build_fcl_circuit(self):
        qc = QuantumCircuit(self.n_qubits)
        theta = qiskit.circuit.Parameter("theta")
        qc.h(range(self.n_qubits))
        qc.ry(theta, range(self.n_qubits))
        qc.measure_all()
        return qc

    def fcl_run(self, theta: float) -> np.ndarray:
        bound = {self.fcl_circ.parameters[0]: theta}
        qc = self.fcl_circ.bind_parameters(bound)
        job = qiskit.execute(qc, self.backend_qiskit, shots=1024)
        result = job.result().get_counts(qc)
        probs = np.array(list(result.values())) / 1024
        states = np.array([int(k, 2) for k in result.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])

    # ------------------------------------------------------------------
    # 2. Self‑attention Qiskit circuit
    # ------------------------------------------------------------------
    def _build_attention_circuit(self):
        qr = QuantumRegister(self.n_qubits, "q")
        cr = ClassicalRegister(self.n_qubits, "c")
        qc = QuantumCircuit(qr, cr)
        self.attn_params = []
        for i in range(self.n_qubits):
            rx = qiskit.circuit.Parameter(f"rx_{i}")
            ry = qiskit.circuit.Parameter(f"ry_{i}")
            rz = qiskit.circuit.Parameter(f"rz_{i}")
            qc.rx(rx, i)
            qc.ry(ry, i)
            qc.rz(rz, i)
            self.attn_params.extend([rx, ry, rz])
        self.attn_entangle_params = []
        for i in range(self.n_qubits - 1):
            crx = qiskit.circuit.Parameter(f"crx_{i}")
            qc.crx(crx, i, i + 1)
            self.attn_entangle_params.append(crx)
        qc.measure_all()
        return qc

    def attn_run(self,
                 rotation_params: np.ndarray,
                 entangle_params: np.ndarray) -> dict:
        bind_dict = {p: val for p, val in zip(self.attn_params, rotation_params)}
        bind_dict.update({p: val for p, val in zip(self.attn_entangle_params, entangle_params)})
        qc = self.attn_circ.bind_parameters(bind_dict)
        job = qiskit.execute(qc, self.backend_qiskit, shots=1024)
        return job.result().get_counts(qc)

    # ------------------------------------------------------------------
    # 3. TorchQuantum RBF kernel
    # ------------------------------------------------------------------
    def _build_kernel(self):
        device = tq.QuantumDevice(n_wires=4)
        kernel = tq.QuantumModule()

        @tq.static_support
        def kernel_forward(qd, x, y):
            qd.reset_states(x.shape[0])
            for i in range(4):
                func_name_dict["ry"](qd, x[:, i], wires=i)
            for i in range(4):
                func_name_dict["ry"](qd, -y[:, i], wires=i)
            return torch.abs(qd.states.view(-1)[0])

        kernel.forward = kernel_forward
        return kernel

    def kernel_run(self, x: np.ndarray, y: np.ndarray) -> float:
        return float(self.kernel(x, y).item())

    # ------------------------------------------------------------------
    # 4. StrawberryFields photonic fraud detection
    # ------------------------------------------------------------------
    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def _apply_sf_layer(self, modes, params, clip: bool):
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

    def sf_build_program(self, params_list: list):
        prog = sf.Program(2)
        with prog.context as q:
            for i, params in enumerate(params_list):
                self._apply_sf_layer(q, params, clip=(i > 0))
        return prog

    def sf_run(self, prog: sf.Program) -> sf.Result:
        return self.backend_sf.run(prog)

__all__ = ["HybridQuantumFCL"]
