import pytest
from app.main import app, settings

def test_settings():
    assert settings is not None
