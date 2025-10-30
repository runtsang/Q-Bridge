import torch
import torchquantum as tq

class QuantumSampler(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 2
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch = inputs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=inputs.device)
        for i in range(batch):
            angle0 = inputs[i, 0]
            angle1 = inputs[i, 1]
            tq.RY(angle0, wires=0)(qdev)
            tq.RY(angle1, wires=1)(qdev)
            tq.CNOT(qdev, wires=[0,1])
        out = self.measure(qdev)
        p0 = (1 + out[:, 0]) / 2
        p1 = 1 - p0
        return torch.stack([p0, p1], dim=1)

class QuantumRegressionHead(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.n_wires = 1
        self.trainable_rot = tq.RZ(has_params=True, trainable=True)
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        batch = probs.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=probs.device)
        for i in range(batch):
            angle = probs[i, 0]
            tq.RY(angle, wires=0)(qdev)
        self.trainable_rot(qdev)
        out = self.measure(qdev)
        return out[:, 0]

class HybridSamplerRegressor(tq.QuantumModule):
    def __init__(self):
        super().__init__()
        self.sampler = QuantumSampler()
        self.regressor = QuantumRegressionHead()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        probs = self.sampler(inputs)
        return self.regressor(probs)
