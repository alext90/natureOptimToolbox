import numpy as np
from base_optimizer import BaseOptimizer
from result import Result

class FireflyAlgorithm(BaseOptimizer):
    def __init__(self, 
                 population,
                 alpha,
                 beta0,
                 gamma,
                 n_generation,
                 error_tol=0.01,
                 verbose=False
                 ):
        super().__init__(population, n_generation, error_tol, verbose)
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
   
    def step(self, t):
        _, best_fitness = self.population.get_best_individual()

        for i in range(self.population.population_size):
            for j in range(self.population.population_size):
                if self.population.fitness[i] > self.population.fitness[j]:  # Move firefly i towards j
                    r = np.linalg.norm(self.population.individuals[i] - self.population.individuals[j])
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                    epsilon = self.alpha * (np.random.uniform(-0.5, 0.5, self.population.dim_individuals))
                    new_solution = self.population.individuals[i] + beta * (self.population.individuals[j] - self.population.individuals[i]) + epsilon
                    new_solution = np.clip(new_solution, self.population.lb, self.population.ub)
                    
                    new_fitness = self.population.objective_function(new_solution)
                    
                    if new_fitness < self.population.fitness[i]:
                        self.population.individuals[i] = new_solution
                        self.population.fitness[i] = new_fitness

                        if new_fitness < best_fitness:
                            best_fitness = new_fitness
