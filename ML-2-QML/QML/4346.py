import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import ParameterVector

class SamplerQNNGen020:
    """
    Quantum counterpart of SamplerQNNGen020 that implements a feature map,
    QCNN‑style convolution and pooling layers, a quantum self‑attention block,
    and a sampler that returns a 2‑bit probability distribution.
    """
    def __init__(self):
        self.backend = Aer.get_backend('qasm_simulator')
        self.num_qubits = 8
        self._build_circuit()

    def _build_circuit(self):
        # Parameter vectors
        self.input_params = ParameterVector('x', self.num_qubits)
        self.weight_params = ParameterVector('w', self.num_qubits * 3)      # conv + pool
        self.rotation_params = ParameterVector('r', self.num_qubits * 3)    # self‑attention rotation
        self.entangle_params = ParameterVector('e', self.num_qubits - 1)    # self‑attention entangle

        # 8 qubits, 2 classical bits for final 2‑bit output
        self.circuit = QuantumCircuit(self.num_qubits, 2)

        # Feature map: simple X rotations encoding input
        for i, p in enumerate(self.input_params):
            self.circuit.rx(p, i)

        # Convolution block (4 pairs)
        for i in range(0, self.num_qubits, 2):
            idx = 3 * (i // 2)
            self._apply_conv_circuit(i, i + 1, self.weight_params[idx:idx + 3])

        # Pooling block (4 pairs)
        offset = self.num_qubits // 2 * 3
        for i in range(0, self.num_qubits, 2):
            idx = offset + 3 * (i // 2)
            self._apply_pool_circuit(i, i + 1, self.weight_params[idx:idx + 3])

        # Quantum self‑attention block
        for i in range(self.num_qubits):
            self.circuit.rx(self.rotation_params[3 * i], i)
            self.circuit.ry(self.rotation_params[3 * i + 1], i)
            self.circuit.rz(self.rotation_params[3 * i + 2], i)
        for i in range(self.num_qubits - 1):
            self.circuit.crx(self.entangle_params[i], i, i + 1)

        # Measure only the last two qubits for a 2‑bit output
        self.circuit.measure(6, 0)
        self.circuit.measure(7, 1)

    def _apply_conv_circuit(self, q1, q2, params):
        """
        Convolution sub‑circuit used in QCNN. Parameters: [rz, ry, cx].
        """
        self.circuit.rz(-np.pi / 2, q2)
        self.circuit.cx(q2, q1)
        self.circuit.rz(params[0], q1)
        self.circuit.ry(params[1], q2)
        self.circuit.cx(q1, q2)
        self.circuit.ry(params[2], q2)
        self.circuit.cx(q2, q1)
        self.circuit.rz(np.pi / 2, q1)

    def _apply_pool_circuit(self, q1, q2, params):
        """
        Pooling sub‑circuit used in QCNN. Parameters: [rz, ry, cx].
        """
        self.circuit.rz(-np.pi / 2, q2)
        self.circuit.cx(q2, q1)
        self.circuit.rz(params[0], q1)
        self.circuit.ry(params[1], q2)
        self.circuit.cx(q1, q2)
        self.circuit.ry(params[2], q2)

    def run(self, input_data, weight_vals, rotation_vals, entangle_vals, shots=1024):
        """
        Execute the quantum circuit with provided parameter values.

        Parameters
        ----------
        input_data : list or array of length 8
            Input angles for the feature map.
        weight_vals : list or array of length 24
            Parameters for convolution and pooling layers.
        rotation_vals : list or array of length 24
            Rotation parameters for the self‑attention block.
        entangle_vals : list or array of length 7
            Entanglement parameters for the self‑attention block.
        shots : int, optional
            Number of shots for the simulator.

        Returns
        -------
        dict
            Probability distribution over 2‑bit strings (e.g., '00', '01').
        """
        param_bindings = {
            **{p: val for p, val in zip(self.input_params, input_data)},
            **{p: val for p, val in zip(self.weight_params, weight_vals)},
            **{p: val for p, val in zip(self.rotation_params, rotation_vals)},
            **{p: val for p, val in zip(self.entangle_params, entangle_vals)},
        }
        bound_circuit = self.circuit.bind_parameters(param_bindings)
        job = execute(bound_circuit, self.backend, shots=shots)
        result = job.result()
        counts = result.get_counts(bound_circuit)
        probs = {k: v / shots for k, v in counts.items()}
        return probs

__all__ = ["SamplerQNNGen020"]
