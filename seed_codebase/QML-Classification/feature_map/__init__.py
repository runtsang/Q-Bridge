"""Lightweight ZZ-feature-map builders and variants.

This package mirrors the structure of :mod:`qiskit.circuit.library` feature maps
but provides small, dependency-light reimplementations suitable for experiments.

* :mod:`.zz_feature_map` – canonical Hadamard + CX–P–CX ZZ feature map.
* :mod:`.zz_feature_map_rzz` – uses native ``rzz`` entanglers for the ZZ phase.
* :mod:`.zz_feature_map_poly` – polynomial single-qubit map with configurable basis.
"""
from .zz_feature_map import ZZFeatureMap, zz_feature_map
from .zz_feature_map_rzz import ZZFeatureMapRZZ, zz_feature_map_rzz
from .zz_feature_map_poly import ZZFeatureMapPoly, zz_feature_map_poly

__all__ = [
    "ZZFeatureMap",
    "zz_feature_map",
    "ZZFeatureMapRZZ",
    "zz_feature_map_rzz",
    "ZZFeatureMapPoly",
    "zz_feature_map_poly",
]
