import numpy as np
from base_optimizer import BaseOptimizer

class WhaleOptimizationAlgorithm(BaseOptimizer):
    def __init__(self, population, 
                 n_generations, 
                 verbose=False, 
                 error_tol: float = 1e-3
                 ):
        '''
        Whale Optimization Algorithm class

        Input:
        - population: Population object
        - n_generations: Number of iterations
        - verbose: Print information about the optimization process
        '''
        super().__init__(population, n_generations, error_tol, verbose)


    def step(self, t):
        best_idx, best_fitness = self.population.get_best_individual()
        best_solution = self.population.individuals[best_idx]
        a = 2 - t * (2 / self.n_generations)
        a2 = -1 + t * (-1 / self.n_generations)
        
        for i in range(self.population.population_size):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = np.random.rand()
            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * best_solution - self.population.individuals[i])
                    self.population.individuals[i] = best_solution - A * D
                else:
                    random_whale = self.population.individuals[np.random.randint(self.population.population_size)]
                    D = np.abs(C * random_whale - self.population.individuals[i])
                    self.population.individuals[i] = random_whale - A * D
            else:
                distance_to_best = np.abs(best_solution - self.population.individuals[i])
                self.population.individuals[i] = distance_to_best * np.exp(a2 * np.random.rand()) * np.cos(2 * np.pi * np.random.rand()) + best_solution
        
        for i in range(self.population.population_size):
            self.population.fitness[i] = self.population.objective_function(self.population.individuals[i])