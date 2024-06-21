import numpy as np

class BatSearch():
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
        self.population = population
        self.velocities = np.zeros((self.population.population_size, self.population.dim_individuals))
        self.r = r
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.f_min = f_min
        self.f_max = f_max
        self.n_generations = n_generations
        self.error_tol = error_tol
        self.verbose = verbose

    # Bat Algorithm (n, dim, lb, ub, f_min, f_max, A, r, alpha, gamma, n_iterations):
    def run(self) -> tuple:
        best_index, best_fitness = self.population.get_best_individual()
        best_solution = self.population.individuals[best_index]

        for t in range(self.n_generations):
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
            
            if self.verbose:
                print(f"Iteration {t+1}, Best fitness: {best_fitness}")

            if best_fitness < self.error_tol:
                print(f"Converged at iteration {t+1}")
                break

        return best_solution, best_fitness
