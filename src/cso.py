import numpy as np

# Levy flight
def levy_flight(Lambda):
    sigma1 = np.power((np.math.gamma(1 + Lambda) * np.sin(np.pi * Lambda / 2)) /
                      (np.math.gamma((1 + Lambda) / 2) * Lambda * np.power(2, (Lambda - 1) / 2)), 1 / Lambda)
    sigma2 = 1
    u = np.random.normal(0, sigma1, size=(dim,))
    v = np.random.normal(0, sigma2, size=(dim,))
    step = u / np.power(np.abs(v), 1 / Lambda)
    return step

# Objective function (example: sphere function)
def objective_function(x):
    return np.sum(x**2)

# Cuckoo Search Algorithm
def cuckoo_search(n, dim, lb, ub, pa, n_iterations):
    # Initialize the population of n nests with random solutions
    nests = np.random.uniform(lb, ub, size=(n, dim))
    fitness = np.array([objective_function(nests[i]) for i in range(n)])
    
    # Find the current best solution
    best_nest_index = np.argmin(fitness)
    best_nest = nests[best_nest_index]
    best_fitness = fitness[best_nest_index]

    for t in range(n_iterations):
        new_nests = np.copy(nests)
        
        # Generate new solutions by Levy flight
        for i in range(n):
            step_size = levy_flight(1.5)
            new_solution = nests[i] + step_size * (nests[i] - best_nest)
            new_solution = np.clip(new_solution, lb, ub)
            new_fitness = objective_function(new_solution)
            
            if new_fitness < fitness[i]:
                new_nests[i] = new_solution
                fitness[i] = new_fitness

        # Discovery and randomization
        for i in range(n):
            if np.random.rand() < pa:
                new_solution = np.random.uniform(lb, ub, size=(dim,))
                new_fitness = objective_function(new_solution)
                
                if new_fitness < fitness[i]:
                    new_nests[i] = new_solution
                    fitness[i] = new_fitness
        
        # Update the nests and best solution
        nests = new_nests
        best_nest_index = np.argmin(fitness)
        best_nest = nests[best_nest_index]
        best_fitness = fitness[best_nest_index]

        print(f"Iteration {t+1}, Best fitness: {best_fitness}")

    return best_nest, best_fitness


