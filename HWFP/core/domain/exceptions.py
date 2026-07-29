from __future__ import annotations


class HWFPError(Exception):
    """Base exception for the HWFP package."""


class DomainValidationError(HWFPError):
    """Raised when a domain entity invariant is violated."""


class StateNotFoundError(HWFPError):
    """Raised when requested match or team state is not found."""


class OddsNotFoundError(HWFPError):
    """Raised when requested odds are not found."""


class RefereeNotFoundError(HWFPError):
    """Raised when a referee profile is not found."""


class ModelInputError(HWFPError):
    """Raised when model input shape or type is invalid."""


class SinkWriteError(HWFPError):
    """Raised when writing a prediction to a sink fails."""


class ModelNotFound(HWFPError):
    """Raised when a requested model_id is not in the registry."""


class NoProductionModelError(HWFPError):
    """Raised when no model is marked as production in the registry."""


class PromotionError(HWFPError):
    """Raised when a candidate model does not improve over production."""


class PromotionGateFailed(HWFPError):
    """Raised when a promotion gates check fails."""


class TrainingDataError(HWFPError):
    """Raised when training data iteration or access fails."""
