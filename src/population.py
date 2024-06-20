import numpy as np

class Population:
    def __init__(self, population_size, dim_individuals, lb, ub, objective_function):
        if not callable(objective_function):
            raise TypeError("objective_function should be a function")
        
        if not isinstance(objective_function(np.zeros(dim_individuals)), (int, float)):
            raise ValueError("objective_function should return a number")
        
        self.population_size = population_size
        self.dim_individuals = dim_individuals
        self.lb = lb
        self.ub = ub
        self.objective_function = objective_function
        self.individuals = np.random.uniform(self.lb, self.ub, size=(self.population_size, self.dim_individuals))
        self.fitness = np.array([self.objective_function(self.individuals[i]) for i in range(self.population_size)])

