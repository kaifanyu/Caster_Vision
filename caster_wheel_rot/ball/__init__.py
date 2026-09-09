"""Approach-A ball-caster pipeline and common-interface adapter."""

from .adapter import adapt_ball_pipeline
from .pipeline import PipelineResult, run_pipeline

__all__ = ["PipelineResult", "adapt_ball_pipeline", "run_pipeline"]
