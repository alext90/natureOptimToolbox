import pytest
'''from src.population import Population
from src.artifical_bee_colony import ArtificalBeeColony
from src.cuckoo_search import CuckooSearch

def objective_function(x):
    return sum(x**2)

pop = Population(
    population_size=50, 
    dim_individuals=30, 
    lb=-1.0, 
    ub=1.0, 
    objective_function=objective_function
)

def test_abc_run():
    abc = ArtificalBeeColony(
        population=pop, 
        n_iter=10000, 
        n_onlookers=10, 
        n_scouts=10, 
        limit=100
    )

    best_solution, best_solution_fitness = abc.run()
    assert best_solution_fitness <= 1e-3

def test_cs_run():
    cs = CuckooSearch(
        population=pop, 
        n_iter=10000, 
        p_discovery=0.25, 
        lambda_levy=1.5
    )

    best_solution, best_solution_fitness = cs.run()
    assert best_solution_fitness <= 1e-3
'''