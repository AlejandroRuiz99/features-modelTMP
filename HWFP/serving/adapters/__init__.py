"""Serving adapter stubs — re-export all stub classes."""

from __future__ import annotations

from HWFP.serving.adapters.codere_odds_adapter import CodereOddsAdapter
from HWFP.serving.adapters.filesystem_model_registry import FilesystemModelRegistry
from HWFP.serving.adapters.inline_overlay_engine import InlineOverlayEngine
from HWFP.serving.adapters.kelly_ev_calculator import KellyEVCalculator
from HWFP.serving.adapters.kelly_staking_calculator import KellyStakingCalculator
from HWFP.serving.adapters.pytorch_feature_builder import PyTorchFeatureBuilder
from HWFP.serving.adapters.pytorch_foul_model import PyTorchFoulModel
from HWFP.serving.adapters.supabase_prediction_sink import SupabasePredictionSink
from HWFP.serving.adapters.supabase_state_adapter import SupabaseStateAdapter
from HWFP.serving.adapters.yaml_referee_profiler import YamlRefereeProfiler

__all__ = [
    "SupabaseStateAdapter",
    "CodereOddsAdapter",
    "PyTorchFeatureBuilder",
    "PyTorchFoulModel",
    "YamlRefereeProfiler",
    "InlineOverlayEngine",
    "KellyEVCalculator",
    "KellyStakingCalculator",
    "SupabasePredictionSink",
    "FilesystemModelRegistry",
]
