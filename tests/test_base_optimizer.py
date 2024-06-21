import pytest
from src.base_optimizer import BaseOptimizer

def test_base_optimizer():
    with pytest.raises(TypeError):
        BaseOptimizer(1, 1.5)
    with pytest.raises(TypeError):
        BaseOptimizer("population", 100)
    