"""Unit tests for the CLI module."""

import pytest
from unittest.mock import patch
from egora.cli import main


def test_version_command(capsys):
    """egora version prints the version string."""
    with patch("sys.argv", ["egora", "version"]):
        main()
    captured = capsys.readouterr()
    assert "egora 0." in captured.out


def test_no_command_shows_help(capsys):
    """egora with no args shows help and exits."""
    with patch("sys.argv", ["egora"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0


def test_diagnose_missing_args():
    """egora diagnose without model args should error."""
    with patch("sys.argv", ["egora", "diagnose"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
