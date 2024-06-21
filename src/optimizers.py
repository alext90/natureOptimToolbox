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
        super().__init__(population, 
                         n_generations, 
                         "BatSearch",
                         error_tol, 
                         verbose)
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


class CuckooSearch(BaseOptimizer):
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
        super().__init__(population, 
                         n_generations, 
                         "CuckooSearch",
                         error_tol, 
                         verbose)
        self.pa = pa
        self.l = lambda_levy_flight

    def levy_flight(self) -> np.array:
        '''Generate a Levy flight'''
        sigma1 = np.power((np.random.gamma(1 + self.l) * np.sin(np.pi * self.l / 2)) / 
                          (np.random.gamma((1 + self.l) / 2) * self.l * np.power(2, (self.l - 1) / 2)), 1 / self.l)
        sigma2 = 1
        u = np.random.normal(0, sigma1, size=(self.population.dim_individuals,))
        v = np.random.normal(0, sigma2, size=(self.population.dim_individuals,))
        step = u / np.power(np.abs(v), 1 / self.l)
        return step   
    
    def step(self, t):
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
        super().__init__(population, 
                         n_generation, 
                         "FireflyAlgorithm",
                         error_tol, 
                         verbose)
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


class WhaleOptimizationAlgorithm(BaseOptimizer):
    def __init__(self, 
                 population, 
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
        super().__init__(population, 
                         n_generations, 
                         "WhaleOptimizationAlgorithm",
                         error_tol, 
                         verbose)


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