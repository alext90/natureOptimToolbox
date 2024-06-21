import pytest
from src.population import Population

def test_incorrect_objective_function():
    with pytest.raises(TypeError):
        _ = Population(
            population_size=50, 
            dim_individuals=30, 
            lb=-1.0, 
            ub=1.0, 
            objective_function="not a function"
        )

    with pytest.raises(ValueError):
        _ = Population(
            population_size=50, 
            dim_individuals=30, 
            lb=-1.0, 
            ub=1.0, 
            objective_function=lambda x: "not a number"
        )

def test_update_individual():
    def objective_function(x):
        return sum(x**2)

    pop = Population(
        population_size=50, 
        dim_individuals=30, 
        lb=-1.0, 
        ub=1.0, 
        objective_function=objective_function
    )

    with pytest.raises(ValueError):
        pop.update_individual("not a number", pop.individuals[0], 0)

    with pytest.raises(TypeError):
        pop.update_individual(1.0, "not an array", 0)

    pop.update_individual(1.0, pop.individuals[0], 0)
    assert pop.fitness[0] == 1.0