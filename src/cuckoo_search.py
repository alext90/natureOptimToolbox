import numpy as np

class CuckooSearch:
    def __init__(self, 
                 population, 
                 pa: float = 0.25, 
                 lambda_levy_flight: float = 1.5, 
                 n_generations: int = 100,
                 error_tol: float = 1e-6,
                 verbose: bool = False,
                 ):
        '''
        Class for cuckoo search optimization
        Input:
        - population: Population object
        - pa: probability of discovery
        - lambda_levy_flight: lambda for levy flight
        - n_generations: number of generations
        - error_tol: error tolerance
        - verbose: print information during optimization     
        '''
        self.population = population
        self.pa = pa # probability of discovery
        self.l = lambda_levy_flight # lambda for levy flight
        self.n_iterations = n_generations
        self.verbose = verbose
        self.error_tol = error_tol

    def levy_flight(self) -> np.array:
        '''Generate a Levy flight'''
        sigma1 = np.power((np.random.gamma(1 + self.l) * np.sin(np.pi * self.l / 2)) / 
                          (np.random.gamma((1 + self.l) / 2) * self.l * np.power(2, (self.l - 1) / 2)), 1 / self.l)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=(self.population.dim_individuals,))
        v = np.random.normal(0, sigma2, size=(self.population.dim_individuals,))
        step = u / np.power(np.abs(v), 1 / self.l)
        return step   
    
    def run(self) -> tuple:
        '''
        Run cuckoo search
        A nest is a solution to the optimization problem and a individual in the population

        Output:
        - best_nest: Best solution found
        - best_fitness: Fitness value of the best solution
        '''       
        # Find the current best solution
        best_nest_index, best_fitness = self.population.get_best_individual()
        best_nest = self.population.individuals[best_nest_index]

        for t in range(self.n_iterations):
            new_nests = np.copy(self.population.individuals)
            
            # Generate new solutions by Levy flight
            for i in range(self.population.population_size):
                step_size = self.levy_flight()
                new_solution = self.population.individuals[i] + step_size * (self.population.individuals[i] - best_nest)
                new_solution = np.clip(new_solution, self.population.lb, self.population.ub)
                new_fitness = self.population.objective_function(new_solution)
                self.population.update_individual(new_fitness, new_solution, i)

            # Discovery and randomization
            for i in range(self.population.population_size):
                if np.random.rand() < self.pa:
                    new_solution = np.random.uniform(self.population.lb, self.population.ub, size=(self.population.dim_individuals,))
                    new_fitness = self.population.objective_function(new_solution)
                    self.population.update_individual(new_fitness, new_solution, i)
            
            # Update the nests and best solution
            nests = new_nests
            best_nest_index, best_fitness = self.population.get_best_individual()
            best_nest = nests[best_nest_index]

            if self.verbose:
                print(f"Iteration {t+1}, Best fitness: {best_fitness}")

            if best_fitness < self.error_tol:
                print(f"Converged at iteration {t+1}")
                break

        return best_nest, best_fitness