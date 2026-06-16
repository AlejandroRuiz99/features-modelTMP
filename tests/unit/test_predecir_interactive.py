"""
Unit tests for predecir_jornada interactive vs non-interactive contract (R12).

Tests written FIRST (TDD RED phase).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from predecir_jornada import RefereeHandlingStrategy, interactive_checkpoint


class TestInteractiveCheckpointNonInteractive:
    """R12: Non-interactive mode skips all prompts."""

    def test_non_interactive_skips_confirmation_prompt(self) -> None:
        """In non-interactive mode, confirmation prompt is never shown."""
        with patch("builtins.input") as mock_input:
            interactive_checkpoint(
                message="Confirm match list?",
                interactive=False,
            )
            mock_input.assert_not_called()

    def test_non_interactive_returns_true(self) -> None:
        """Non-interactive checkpoint always returns True (proceed)."""
        result = interactive_checkpoint(
            message="Continue?",
            interactive=False,
        )
        assert result is True

    def test_interactive_calls_input(self) -> None:
        """Interactive mode calls input() for user confirmation."""
        with patch("builtins.input", return_value="y") as mock_input:
            interactive_checkpoint(
                message="Continue?",
                interactive=True,
            )
            mock_input.assert_called_once()

    def test_interactive_returns_false_on_no(self) -> None:
        """Interactive mode returns False when user says no."""
        with patch("builtins.input", return_value="n"):
            result = interactive_checkpoint(
                message="Continue?",
                interactive=True,
            )
        assert result is False

    def test_interactive_returns_true_on_yes(self) -> None:
        """Interactive mode returns True when user says yes."""
        with patch("builtins.input", return_value="y"):
            result = interactive_checkpoint(
                message="Continue?",
                interactive=True,
            )
        assert result is True


class TestNullRefereeHandling:
    """R12: Non-interactive + null referee proceeds without abort."""

    def test_auto_run_without_strategy_in_non_interactive(self) -> None:
        """Non-interactive mode resolves to AUTO_RUN_WITHOUT strategy."""
        strategy = RefereeHandlingStrategy.from_interactive(interactive=False)
        assert strategy == RefereeHandlingStrategy.AUTO_RUN_WITHOUT

    def test_interactive_strategy_is_prompt(self) -> None:
        """Interactive mode resolves to PROMPT strategy."""
        strategy = RefereeHandlingStrategy.from_interactive(interactive=True)
        assert strategy == RefereeHandlingStrategy.PROMPT

    def test_auto_run_without_does_not_abort(self) -> None:
        """AUTO_RUN_WITHOUT strategy does not raise on null referee."""
        # Simulated: non-interactive pipeline continues with null referee
        strategy = RefereeHandlingStrategy.AUTO_RUN_WITHOUT
        # The strategy should not raise when proceeding with null referees
        try:
            strategy.handle_null_referee(match_name="Sevilla vs Getafe")
        except SystemExit:
            pytest.fail("AUTO_RUN_WITHOUT should not abort on null referee")

    def test_auto_run_without_returns_none(self) -> None:
        """AUTO_RUN_WITHOUT handle_null_referee returns None (no abort)."""
        strategy = RefereeHandlingStrategy.AUTO_RUN_WITHOUT
        result = strategy.handle_null_referee(match_name="Real Madrid vs Barcelona")
        # Returns None or a log message — does NOT raise
        assert result is None or isinstance(result, str)


class TestCriticalFailureAlwaysAborts:
    """R12: Critical subprocess failures always abort regardless of mode."""

    def test_subprocess_failure_raises_in_non_interactive(self) -> None:
        """Critical subprocess error aborts even in non-interactive mode."""
        from predecir_jornada import handle_subprocess_error

        with pytest.raises((RuntimeError, SystemExit, ValueError)):
            handle_subprocess_error(
                returncode=1,
                stderr="Fatal error in update_stats.py",
                context="update_stats",
                interactive=False,
            )

    def test_subprocess_failure_raises_in_interactive(self) -> None:
        """Critical subprocess error aborts even in interactive mode."""
        from predecir_jornada import handle_subprocess_error

        with pytest.raises((RuntimeError, SystemExit, ValueError)):
            handle_subprocess_error(
                returncode=1,
                stderr="Fatal error in run_prediction.py",
                context="run_prediction",
                interactive=True,
            )

    def test_zero_exit_does_not_raise(self) -> None:
        """Zero exit code does not raise an error."""
        from predecir_jornada import handle_subprocess_error

        # Should not raise for successful subprocess
        result = handle_subprocess_error(
            returncode=0,
            stderr="",
            context="update_stats",
            interactive=False,
        )
        # Returns None or similar for success
        assert result is None
