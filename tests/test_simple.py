"""Simple test for validation"""
import pytest
from yetanotherspdnet import hello, __version__


def test_hello():
    """Test the hello function."""
    result = hello()
    assert result == "Hello from YetAnotherSPDNet!"
    assert isinstance(result, str)


def test_version():
    """Test version is defined."""
    assert __version__ is not None
    assert isinstance(__version__, str)
    assert len(__version__) > 0
