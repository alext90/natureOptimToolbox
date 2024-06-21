from abc import ABC, abstractmethod
import numpy as np
from src.population import Population
from src.result import Result

class BaseOptimizer(ABC):
    def __init__(self, 
                population: Population, 
                n_generations: int, 
                error_tol=1e-6, 
                verbose=False):
        
        if not isinstance(population, Population):
            raise ValueError("population must be an instance of Population")

        if not isinstance(n_generations, int):
            raise ValueError("n_generations must be an integer")

        self.population = population
        self.n_generations = n_generations
        self.error_tol = error_tol
        self.verbose = verbose

    @abstractmethod
    def step(self, gen_num):
        """Perform a single optimization step and update the current solution."""
        pass

    def run(self):
        """Run the optimizer for a given number of iterations."""
        fitness_history = np.ndarray((self.n_generations, self.population.fitness.size))
        for t in range(self.n_generations):
            self.step(t)

            best_idx, best_fitness = self.population.get_best_individual()
            best_solution = self.population.individuals[best_idx]

            if self.verbose:    
                print(f"Iteration {t+1}, Best fitness: {best_fitness}")
        
            if best_fitness < self.error_tol:
                print(f"Converged at iteration {t+1}")
                fitness_history = fitness_history[:t+1, :]
                break

        result = Result(best_solution, 
                        best_fitness, 
                        t+1, 
                        self.population, 
                        fitness_history)

        return result