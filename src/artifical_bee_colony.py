import numpy as np

class ArtificialBeeColony:
    def __init__(self, population_size: int, 
                objective_function: callable, 
                lb: float, 
                ub: float, 
                dim: int, 
                limit: int,
                n_generations: int = 100, 
                ) -> None:
        self.population_size = population_size
        self.lb = lb
        self.ub = ub
        self.n_generations = n_generations
        self.dim = dim
        self.limit = limit
        self.objective_function = objective_function
        self.fitness = None
        self.population = None

    def calc_new_inidividual(self, i, k):
        '''Calculate new individual using the formula: x_new = x + phi * (x - x_k)'''
        phi = np.random.uniform(-1, 1, size=(self.dim,))
        new_solution = self.population[i] + phi * (self.population[i] - self.population[k])
        new_solution = np.clip(new_solution, self.lb, self.ub)
        new_fitness = self.objective_function(new_solution)
        return new_fitness, new_solution

    def update_individual(self, new_fitness, new_solution, i):
        '''Update individual if new fitness is better than the current fitness'''
        if new_fitness < self.fitness[i]:
            self.population[i] = new_solution
            self.fitness[i] = new_fitness

    def employed_bees_phase(self):
        '''Employed Bees Phase'''
        for i in range(len(self.population)):
            k = np.random.randint(0, len(self.population))
            while k == i:
                k = np.random.randint(0, len(self.population))
            
            new_fitness, new_solution = self.calc_new_inidividual(i, k)
            self.update_individual(new_fitness, new_solution, i)

    def onlooker_bees_phase(self):
        '''Onlooker Bees Phase'''
        new_population = np.copy(self.population)
        fitness_prob = self.fitness / np.sum(self.fitness)
        
        for _ in range(len(self.population)):
            i = np.random.choice(len(self.population), p=fitness_prob)
            
            k = np.random.randint(0, len(self.population))
            while k == i:
                k = np.random.randint(0, len(self.population))
            
            new_fitness, new_solution = self.calc_new_inidividual(i, k)
            self.update_individual(new_fitness, new_solution, i)

        self.population = new_population

    
    def scout_bees_phase(self):
        '''Scout Bees Phase'''
        new_population = np.copy(self.population)
        new_fitness = np.copy(self.fitness)
        
        for i in range(len(self.population)):
            if self.fitness[i] > self.limit:
                new_solution = np.random.uniform(self.lb, self.ub, size=(self.dim,))
                new_population[i] = new_solution
                new_fitness[i] = self.objective_function(new_solution)
        
        self.population = new_population
        self.fitness = new_fitness


    def run(self) -> tuple:
        self.population = np.random.uniform(self.lb, self.ub, size=(self.population_size, self.dim))
        self.fitness = np.array([self.objective_function(self.population[i]) for i in range(self.population_size)])
        
        for t in range(self.n_generations):
            self.employed_bees_phase()
            self.onlooker_bees_phase()
            self.scout_bees_phase()
            
            # Find the best solution
            best_fitness = np.min(self.fitness)
            best_solution = self.population[np.argmin(self.fitness)]
            
            print(f"Iteration {t+1}, Best fitness: {best_fitness}")
        
        return best_solution, best_fitness