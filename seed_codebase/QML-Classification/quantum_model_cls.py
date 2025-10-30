# quantum_model_cls.py
"""
QML classifier using your custom feature/ansatz variants.

- Prefers VQC (if available in qiskit-ml). Otherwise falls back to a SamplerQNN-based
  NeuralNetworkClassifier (binary) and we build One-vs-Rest in validate_cls.py when needed.
"""

from __future__ import annotations
from typing import Optional

# your circuit variants
from feature_map_variations import ZZFeatureMap, ZZFeatureMapRZZ, ZZFeatureMapPoly
from ansatz_variations import RealAmplitudes, RealAmplitudesAlternating, RealAmplitudesCZ

from qiskit.primitives import Sampler
from qiskit_algorithms.optimizers import (
    COBYLA, L_BFGS_B, SPSA, ESCH, ISRES, DIRECT_L_RAND, CRS, ADAM, CG, POWELL
)

# Prefer VQC when available
_HAS_VQC = True
try:
    from qiskit_machine_learning.algorithms.classifiers import VQC
except Exception:
    _HAS_VQC = False

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

# -------- defaults --------
DEFAULT_FEATURE_MAP = "ZZ"     # "ZZ", "ZZRZZ", "ZZPOLY"
DEFAULT_ANSATZ      = "RA"     # "RA", "RAALT", "RACZ"

FEATURE_MAPS = {
    "ZZ": ZZFeatureMap,
    "ZZRZZ": ZZFeatureMapRZZ,
    "ZZPOLY": ZZFeatureMapPoly,
}
ANSAETZE = {
    "RA": RealAmplitudes,
    "RAALT": RealAmplitudesAlternating,
    "RACZ": RealAmplitudesCZ,
}

# -------- optimizers --------
def spsa_callback(nfev, params, value, stepsize, accepted):
    if not hasattr(spsa_callback, "history"):
        spsa_callback.history = []
    spsa_callback.history.append(float(value))
    return False

def scipy_callback(xk):
    import numpy as np
    if not hasattr(scipy_callback, "history"):
        scipy_callback.history = []
    scipy_callback.history.append(float((xk**2).sum() ** 0.5))
    return False

def get_optimizer(name: str):
    name = (name or "COBYLA").upper()
    if name == "SPSA":
        return SPSA(maxiter=300, learning_rate=0.01, perturbation=0.1, callback=spsa_callback)
    if name == "L_BFGS_B":
        return L_BFGS_B(maxfun=200, callback=scipy_callback)
    if name == "ESCH":
        return ESCH()
    if name == "ISRES":
        return ISRES()
    if name in ("DIRECT_L_RAND", "DIRECT_LRAND", "DIRECT"):
        return DIRECT_L_RAND()
    if name == "CRS":
        return CRS()
    if name == "ADAM":
        return ADAM()
    if name == "CG":
        return CG()
    if name == "POWELL":
        return POWELL(callback=scipy_callback)
    return COBYLA(maxiter=600, callback=scipy_callback)

# -------- builder --------
def build_classifier(
    n_features: int,
    num_reps: int = 2,
    optimizer_name: str = "COBYLA",
    feature_map_name: Optional[str] = None,
    ansatz_name: Optional[str] = None,
):
    fm_key = (feature_map_name or DEFAULT_FEATURE_MAP).upper()
    an_key = (ansatz_name or DEFAULT_ANSATZ).upper()
    if fm_key not in FEATURE_MAPS:
        raise ValueError(f"Unknown feature_map_name {fm_key}. Choose from {list(FEATURE_MAPS.keys())}")
    if an_key not in ANSAETZE:
        raise ValueError(f"Unknown ansatz_name {an_key}. Choose from {list(ANSAETZE.keys())}")

    fm = FEATURE_MAPS[fm_key](feature_dimension=n_features, reps=num_reps, entanglement="linear")
    an = ANSAETZE[an_key](n_features, reps=num_reps, entanglement="linear")
    opt = get_optimizer(optimizer_name)
    sampler = Sampler()

    if _HAS_VQC:
        # VQC usually handles binary directly; for >2 classes we wrap in One-vs-Rest in validate_cls.py
        return VQC(feature_map=fm, ansatz=an, optimizer=opt, sampler=sampler)

    # Fallback: binary classifier via SamplerQNN + parity
    circuit = fm.compose(an)

    def parity(x):
        return x % 2

    qnn = SamplerQNN(
        circuit=circuit,
        input_params=list(fm.parameters),
        weight_params=list(an.parameters),
        sampler=sampler,
        interpret=parity,
        output_shape=2,
    )
    clf = NeuralNetworkClassifier(qnn, optimizer=opt)
    return clf
