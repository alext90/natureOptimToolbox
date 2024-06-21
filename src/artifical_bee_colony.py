import numpy as np
from base_optimizer import BaseOptimizer
from population import Population

class ArtificialBeeColony(BaseOptimizer):
    def __init__(self, population: Population,
                limit: int,
                n_generations: int = 100, 
                error_tol: float = 1e-6,
                verbose: bool = False,
                ) -> None:
        '''
        Artificial Bee Colony Optimization
        Input:
        - population: Population object
        - limit: limit for the number of trials
        - n_generations: number of generations
        - error_tol: error tolerance
        - verbose: print information during optimization
        '''
        super().__init__(population, 
                         n_generations, 
                         "ArtificialBeeColony",
                         error_tol, 
                         verbose)
        self.limit = limit

    def calc_new_inidividual(self, i: int, k: int):
        '''Calculate new individual using the formula: x_new = x + phi * (x - x_k)'''
        phi = np.random.uniform(-1, 1, size=(self.population.dim_individuals,))
        new_solution = self.population.individuals[i] + phi * (self.population.individuals[i] - self.population.individuals[k])
        new_solution = np.clip(new_solution, self.population.lb, self.population.ub)
        new_fitness = self.population.objective_function(new_solution)
        return new_fitness, new_solution

    def employed_bees_phase(self):
        '''Employed Bees Phase'''
        for i in range(self.population.dim_individuals):
            k = np.random.randint(0, self.population.dim_individuals)
            while k == i:
                k = np.random.randint(0, self.population.dim_individuals)
            
            new_fitness, new_solution = self.calc_new_inidividual(i, k)
            self.population.update_individual(new_fitness, new_solution, i)

    def onlooker_bees_phase(self):
        '''Onlooker Bees Phase'''
        fitness_prob = self.population.fitness / np.sum(self.population.fitness)
        
        for _ in range(self.population.dim_individuals):
            i = np.random.choice(self.population.population_size, p=fitness_prob)
            
            k = np.random.randint(0, self.population.dim_individuals)
            while k == i:
                k = np.random.randint(0, self.population.dim_individuals)
            
            new_fitness, new_solution = self.calc_new_inidividual(i, k)
            self.population.update_individual(new_fitness, new_solution, i)
    
    def scout_bees_phase(self):
        '''Scout Bees Phase'''
        new_population = np.copy(self.population.individuals)
        new_fitness = np.copy(self.population.fitness)
        
        for i in range(self.population.dim_individuals):
            if self.population.fitness[i] > self.limit:
                new_solution = np.random.uniform(self.population.lb, self.population.ub,
                                                 size=(self.population.dim_individuals,))
                new_population[i] = new_solution
                new_fitness[i] = self.population.objective_function(new_solution)
        
        self.population.individuals = new_population
        self.population.fitness = new_fitness


    def step(self, t):
        '''Optimization step'''
        self.employed_bees_phase()
        self.onlooker_bees_phase()
        self.scout_bees_phase()