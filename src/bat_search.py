import numpy as np
from base_optimizer import BaseOptimizer
from result import Result

class BatSearch(BaseOptimizer):
    def __init__(self, 
                 population, 
                 r, 
                 A, 
                 alpha, 
                 gamma, 
                 f_min,
                 f_max,
                 n_generations, 
                 error_tol=0.01, 
                 verbose=False
                 ):
        '''
        Bat search algorithm class

        Input:
        - population: Population object
        - velocities: Velocities of the individuals
        - r: Pulse rate
        - A: Loudness
        - alpha: Loudness decay
        - gamma: Pulse rate decay
        - f_min: Minimum frequency
        - f_max: Maximum frequency
        - n_generations: Number of generations
        - error_tol: Error tolerance
        - verbose: Print information about the optimization process
        '''
        super().__init__(population, n_generations, error_tol, verbose)
        self.velocities = np.zeros((self.population.population_size, self.population.dim_individuals))
        self.r = r
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max

    def step(self, t):
        best_index, best_fitness = self.population.get_best_individual()
        best_solution = self.population.individuals[best_index]
        for i in range(self.population.population_size):
            beta = np.random.uniform(0, 1)
            frequency = self.f_min + (self.f_max - self.f_min) * beta
            self.velocities[i] = self.velocities[i] + (self.population.individuals[i] - best_solution) * frequency
            new_solution = self.population.individuals[i] + self.velocities[i]
            new_solution = np.clip(new_solution, self.population.lb, self.population.ub)
            
            if np.random.rand() > self.r:
                epsilon = np.random.uniform(-1, 1, self.population.dim_individuals)
                new_solution = best_solution + epsilon * self.A
            
            new_fitness = self.population.objective_function(new_solution)
            
            if (new_fitness < self.population.fitness[i]) and (np.random.rand() < self.A):
                self.population.individuals[i] = new_solution
                self.population.fitness[i] = new_fitness
                
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness

        self.A = self.A * self.alpha
        self.r = self.r * (1 - np.exp(-self.gamma * t))
