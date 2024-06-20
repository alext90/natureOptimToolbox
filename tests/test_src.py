import pytest
from src.artifical_bee_colony import ArtificialBeeColony

def test_incorrect_objective_function():
    with pytest.raises(TypeError):
        abc = ArtificialBeeColony(
            population_size=50, 
            objective_function="not a function", 
            lb=-1.0, 
            ub=1.0, 
            dim=30, 
            limit=100, 
            n_generations=5000
        )

    with pytest.raises(ValueError):
        abc = ArtificialBeeColony(
            population_size=50, 
            objective_function=lambda x: "not a number", 
            lb=-1.0, 
            ub=1.0, 
            dim=30, 
            limit=100, 
            n_generations=5000
       )