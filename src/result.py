import numpy as np
import matplotlib.pyplot as plt
from population import Population

class Result:
    def __init__(self, 
                 best_solution: np.ndarray, 
                 best_fitness: float, 
                 n_iteration: int, 
                 population: Population,
                 fitness_history: np.ndarray = None,
                 individial_history: np.ndarray = None,
                 algorithm_name: str = None,
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
        self.individual_history = individial_history
        self.algorithm_name = algorithm_name

    def plot_phenotypic_diversity(self):
        '''Plot the fitness history of the algorithm'''
        x = range(self.n_iteration)
        y = np.median(self.fitness_history, axis=1)
        yerr = np.std(self.fitness_history, axis=1)
        plt.errorbar(x, y, yerr=yerr, fmt='-o', color='black')
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.title(f"Phenotypic Diversity\n{self.algorithm_name}")
        plt.grid()
        plt.show()

    def plot_genotypic_diversity(self):
        '''Plot the genotypic diversity'''
        x = range(self.n_iteration)
        y = np.median(np.median(self.individual_history, axis=1), axis=1)
        yerr = np.std(np.std(self.individual_history, axis=1), axis=1)
        plt.errorbar(x, y, yerr=yerr, fmt='-o', color='black')
        plt.xlabel("Generation")
        plt.ylabel("Mean Fitness")
        plt.title(f"Genotypic Diversity\n{self.algorithm_name}")
        plt.grid()
        plt.show()