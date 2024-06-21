import pytest
import numpy as np
from src.result import Result

def test_result_inputs():
    with pytest.raises(ValueError):
        Result("best_solution", 1, 1, None)
    with pytest.raises(ValueError):
        Result(np.array([1, 2]), "best_fitness", 1, None)
    with pytest.raises(ValueError):
        Result(np.array([1, 2]), 1, "n_iteration", None)