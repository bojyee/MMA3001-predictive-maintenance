"""Testable implementation of the frozen MMA3001 AI4I failure classifier.

These modules reproduce the choices in Notebook 06. The six notebooks remain
independently executable and do not import this package.
"""

from .model import FINAL_THRESHOLD, RANDOM_STATE, create_final_model, predict_failure

__all__ = ["FINAL_THRESHOLD", "RANDOM_STATE", "create_final_model", "predict_failure"]
