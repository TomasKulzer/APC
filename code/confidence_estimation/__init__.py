"""Confidence estimation module for model calibration."""

from .temperature_scaling import (
    temperature_scaling,
    find_optimal_temperature,
    get_logits_from_model,
    calibrate_model_temperature,
    evaluate_calibration,
    apply_temperature_scaling
)

from .visualization import plot_reliability_diagram

__all__ = [
    'temperature_scaling',
    'find_optimal_temperature',
    'get_logits_from_model',
    'calibrate_model_temperature',
    'evaluate_calibration',
    'apply_temperature_scaling',
    'plot_reliability_diagram'
]
