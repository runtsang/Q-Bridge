import torch
import torchquantum as tq
import numpy as np
from typing import List, Dict, Sequence
from torchquantum.functional import func_name_dict, op_name_dict

class KernalAnsatz(tq.QuantumModule):
    """
    Variational ansatz built from a user‑supplied list of gate dictionaries.
    Each dictionary must contain:
        - input_idx: list of feature indices used as rotation parameters
        - func: name of the TorchQuantum gate (e.g. 'ry', 'cx')
        - wires: list of target wires
    """
    def __init__(self, func_list: List[Dict]):
        super().__init__()
        self.func_list = func_list

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        q_device.reset_states(x.shape[0])
        for info in self.func_list:
            params = x[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)
        for info in reversed(self.func_list):
            params = -y[:, info["input_idx"]] if op_name_dict[info["func"]].num_params else None
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

class QuantumKernelMethod(tq.QuantumModule):
    """
    Quantum kernel using a flexible variational ansatz.
    Supports GPU evaluation, shot‑noise modeling, and optional Qiskit export.
    """
    def __init__(self,
                 n_wires: int = 4,
                 func_list: List[Dict] | None = None,
                 device: str | torch.device | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.device = torch.device(device) if device else torch.device('cpu')
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires, device=self.device)
        if func_list is None:
            func_list = [{"input_idx": [i], "func": "ry", "wires": [i]} for i in range(n_wires)]
        self.ansatz = KernalAnsatz(func_list)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the overlap |<0|U(x)†U(y)|0>| between two input vectors.
        """
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self,
                      a: Sequence[torch.Tensor],
                      b: Sequence[torch.Tensor],
                      shots: int | None = None,
                      seed: int | None = None) -> np.ndarray:
        """
        Compute the Gram matrix between two datasets.
        When *shots* > 0, Gaussian shot noise with variance 1/shots is added
        to each kernel entry to emulate finite‑sample fluctuations.
        """
        mat = np.array([[self.forward(torch.as_tensor(x), torch.as_tensor(y)).item()
                         for y in b] for x in a])
        if shots is not None and shots > 0:
            rng = np.random.default_rng(seed)
            noise = rng.normal(loc=0.0,
                               scale=1.0/np.sqrt(shots),
                               size=mat.shape)
            mat = np.clip(mat + noise, 0.0, 1.0)
        return mat

    def to_qiskit_circuit(self) -> "qiskit.circuit.QuantumCircuit":
        """
        Export the underlying ansatz as a Qiskit QuantumCircuit for further analysis.
        Requires qiskit to be installed.
        """
        try:
            from qiskit import QuantumCircuit
            qc = QuantumCircuit(self.n_wires)
            for info in self.ansatz.func_list:
                gate = getattr(QuantumCircuit, info["func"])
                if op_name_dict[info["func"]].num_params:
                    qc.append(gate(len(info["input_idx"])), info["wires"])
                else:
                    qc.append(gate(), info["wires"])
            return qc
        except Exception as exc:
            raise RuntimeError("Qiskit is required for circuit export") from exc

__all__ = ["QuantumKernelMethod", "KernalAnsatz"]
