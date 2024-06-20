import pytest
from src.main import greetings

def test_greetings():
    assert greetings("Alex") == "Hello Alex!"
    with pytest.raises(TypeError):
        greetings(123)