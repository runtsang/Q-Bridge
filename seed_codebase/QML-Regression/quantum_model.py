# quantum_model.py
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorEstimator as Estimator
from qiskit_machine_learning.algorithms.regressors import VQR
from qiskit_algorithms.optimizers import (
    COBYLA, L_BFGS_B, SPSA, ESCH, ISRES, DIRECT_L_RAND, CRS, ADAM, CG, POWELL
)

# --- optional callbacks (useful for traces) ---
def spsa_callback(nfev, params, value, stepsize, accepted):
    if not hasattr(spsa_callback, "history"):
        spsa_callback.history = []
    spsa_callback.history.append(float(value))
    return False

def scipy_callback(xk):
    # track a simple proxy across iterations (norm of params)
    import numpy as np
    if not hasattr(scipy_callback, "history"):
        scipy_callback.history = []
    scipy_callback.history.append(float((xk**2).sum() ** 0.5))
    return False

def get_optimizer(name: str):
    name = (name or "COBYLA").upper()
    if name == "SPSA":
        return SPSA(maxiter=500, learning_rate=0.01, perturbation=0.1, callback=spsa_callback)
    if name == "L_BFGS_B":
        return L_BFGS_B(maxfun=300, callback=scipy_callback)
    if name == "ESCH":
        return ESCH()
    if name == "ISRES":
        return ISRES()
    if name == "DIRECT_L_RAND":
        return DIRECT_L_RAND()
    if name == "CRS":
        return CRS()
    if name == "ADAM":
        return ADAM()
    if name == "CG":
        return CG()
    if name == "POWELL":
        return POWELL(callback=scipy_callback)
    # default
    return COBYLA(maxiter=1000, callback=scipy_callback)

def build_vqr(n_features: int, num_reps: int = 3, optimizer_name: str = "COBYLA") -> VQR:
    """Return a VQR configured for a given feature dimension."""
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=num_reps)
    ansatz = RealAmplitudes(n_features, reps=num_reps, entanglement="linear")
    optimizer = get_optimizer(optimizer_name)
    vqr = VQR(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        estimator=Estimator(),  # noiseless statevector
    )
    return vqr
