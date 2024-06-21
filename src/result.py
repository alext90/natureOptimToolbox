import numpy as np
from population import Population

class Result:
    def __init__(self, 
                 best_solution: np.ndarray, 
                 best_fitness: float, 
                 n_iteration: int, 
                 population: Population,
                 fitness_history: np.ndarray = None,
                 ):
        if not isinstance(n_iteration, int):
            raise ValueError("n_iteration must be an integer")
        
        if not isinstance(best_fitness, (int, float)):
            raise ValueError("best_fitness must be an integer or float")
        
        if not isinstance(best_solution, np.ndarray):
            raise ValueError("best_solution must be a numpy array")

        self.best_solution = best_solution
        self.best_fitness = best_fitness
        self.n_iteration = n_iteration
        self.population = population
        self.fitness_history = fitness_history