"""Smoke tests for the Streamlit frontend (app.py).

These use Streamlit's official AppTest harness to execute the app script
headlessly and assert it renders without exceptions. Skipped automatically
when streamlit is not installed.
"""

from pathlib import Path

import pytest

streamlit = pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[2] / "app.py"


def test_app_renders_without_errors() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 5


def test_app_shows_library_count() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    metrics = [m.value for m in at.metric]
    assert any(m == "15" for m in metrics)