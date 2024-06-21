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
        self.fitness = np.array([self.objective_function(ind) for ind in self.individuals])

    def update_individual(self, new_fitness, new_solution: np.ndarray, i: int) -> None:
        '''Update individual if new fitness is better than the current fitness'''
        if not isinstance(new_solution, np.ndarray):
            raise TypeError('new_solution must be a numpy.ndarray')
        
        if not isinstance(new_fitness, (int, float)):
            raise ValueError('new_fitness must be a number')

        if new_fitness < self.fitness[i]:
            self.individuals[i] = new_solution
            self.fitness[i] = new_fitness

    def get_best_individual(self) -> tuple:
        '''Return the best individual and its fitness'''
        best_index = np.argmin(self.fitness)
        return best_index, self.fitness[best_index]