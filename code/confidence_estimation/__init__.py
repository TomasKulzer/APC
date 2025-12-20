"""Confidence estimation module for model calibration."""

from .temperature_scaling import (
    temperature_scaling,
    find_optimal_temperature,
    get_logits_from_model,
    calibrate_model_temperature,
    evaluate_calibration,
    apply_temperature_scaling
)

from .platt_scaling import (
    PlattScaler,
    platt_scale,
    calibrate_model_platt,
    apply_platt_scaling
)

from .isotonic_calibration import (
    IsotonicCalibrator,
    isotonic_calibrate,
    calibrate_model_isotonic,
    apply_isotonic_calibration
)

from .ordinal_calibration import (
    OrdinalCalibrator,
    ordinal_calibrate,
    calibrate_model_ordinal,
    apply_ordinal_calibration,
    verify_ordinal_constraints,
    probs_to_cumulative,
    cumulative_to_probs
)

from .visualization import plot_reliability_diagram

__all__ = [
    'temperature_scaling',
    'find_optimal_temperature',
    'get_logits_from_model',
    'calibrate_model_temperature',
    'evaluate_calibration',
    'apply_temperature_scaling',
    'PlattScaler',
    'platt_scale',
    'calibrate_model_platt',
    'apply_platt_scaling',
    'IsotonicCalibrator',
    'isotonic_calibrate',
    'calibrate_model_isotonic',
    'apply_isotonic_calibration',
    'OrdinalCalibrator',
    'ordinal_calibrate',
    'calibrate_model_ordinal',
    'apply_ordinal_calibration',
    'verify_ordinal_constraints',
    'probs_to_cumulative',
    'cumulative_to_probs',
    'plot_reliability_diagram'
]
