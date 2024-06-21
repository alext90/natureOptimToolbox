import pytest
import numpy as np
from src.result import Result

def test_result_inputs():
    with pytest.raises(ValueError):
        Result("best_solution", 1, 1)
        Result(np.array([1, 2, 3]), "best_fitness", 1)
        Result(np.array([1, 2, 3]), 1, "n_iteration")
        Result(np.array([1, 2, 3]), 1, 1.0)
        Result(np.array([1, 2, 3]), 1.0, 1)
        Result(np.array([1, 2, 3]), 1, 1)