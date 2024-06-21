import numpy as np
from base_optimizer import BaseOptimizer

class GrayWolfOptimizer(BaseOptimizer):
    def __init__(self, population, 
                 n_generations, 
                 verbose=False, 
                 error_tol: float = 1e-3
                 ):
        super().__init__(population, 
                         n_generations, 
                         "GrayWolfOptimizer", 
                         error_tol, 
                         verbose)
        self.n_generations = n_generations
        self.alpha_pos = np.random.uniform(self.population.lb, self.population.ub, self.population.dim_individuals)
        self.alpha_score = self.population.objective_function(self.alpha_pos)
        self.beta_pos = np.random.uniform(self.population.lb, self.population.ub, self.population.dim_individuals)
        self.beta_score = self.population.objective_function(self.beta_pos)
        self.delta_pos = np.random.uniform(self.population.lb, self.population.ub, self.population.dim_individuals)
        self.delta_score = self.population.objective_function(self.delta_pos)

    def step(self, t):
        for i in range(self.population.population_size):
            self.population.fitness[i] = self.population.objective_function(self.population.individuals[i])
            
            if self.population.fitness[i] < self.alpha_score:
                self.alpha_score = self.population.fitness[i]
                self.alpha_pos = self.population.individuals[i]
            elif self.population.fitness[i] < self.beta_score:
                self.beta_score = self.population.fitness[i]
                self.beta_pos = self.population.individuals[i]
            elif self.population.fitness[i] < self.delta_score:
                self.delta_score = self.population.fitness[i]
                self.delta_pos = self.population.individuals[i]
        
        a = 2 - t * (2 / self.n_generations)
        
        for i in range(self.population.population_size):
            r1 = np.random.rand(self.population.dim_individuals)
            r2 = np.random.rand(self.population.dim_individuals)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha_pos - self.population.individuals[i])
            X1 = self.alpha_pos - A1 * D_alpha
            
            r1 = np.random.rand(self.population.dim_individuals)
            r2 = np.random.rand(self.population.dim_individuals)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta_pos - self.population.individuals[i])
            X2 = self.beta_pos - A2 * D_beta
            
            r1 = np.random.rand(self.population.dim_individuals)
            r2 = np.random.rand(self.population.dim_individuals)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta_pos - self.population.individuals[i])
            X3 = self.delta_pos - A3 * D_delta
            
            self.population.individuals[i] = (X1 + X2 + X3) / 3