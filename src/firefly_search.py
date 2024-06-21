import numpy as np

class FireflyAlgorithm:
    def __init__(self, 
                 population,
                 alpha,
                 beta0,
                 gamma,
                 n_generation,
                 error_tol=0.01,
                 verbose=False
                 ):
        self.population = population
        self.alpha = alpha
        self.beta0 = beta0
        self.gamma = gamma
        self.n_generation = n_generation
        self.error_tol = error_tol
        self.verbose = verbose
   
    def run(self):
        best_index, best_fitness = self.population.get_best_individual()
        best_solution = self.population.individuals[best_index]

        for t in range(self.n_generation):
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
                                best_solution = new_solution
                                best_fitness = new_fitness

            if self.verbose:
                print(f"Iteration {t+1}, Best fitness: {best_fitness}")

            if best_fitness < self.error_tol:
                print(f"Converged at iteration {t+1}")
                break

        return best_solution, best_fitness