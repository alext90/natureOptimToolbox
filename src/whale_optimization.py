import numpy as np

def objective_function(x):
    return np.sum(x**2)

def whale_optimization_algorithm(n, dim, lb, ub, n_iterations):
    whales = np.random.uniform(lb, ub, (n, dim))
    fitness = np.array([objective_function(whale) for whale in whales])
    
    best_index = np.argmin(fitness)
    best_solution = whales[best_index]
    best_fitness = fitness[best_index]

    for t in range(n_iterations):
        a = 2 - t * (2 / n_iterations)
        a2 = -1 + t * (-1 / n_iterations)
        
        for i in range(n):
            r1, r2 = np.random.rand(), np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            
            p = np.random.rand()
            if p < 0.5:
                if np.abs(A) < 1:
                    D = np.abs(C * best_solution - whales[i])
                    whales[i] = best_solution - A * D
                else:
                    random_whale = whales[np.random.randint(n)]
                    D = np.abs(C * random_whale - whales[i])
                    whales[i] = random_whale - A * D
            else:
                distance_to_best = np.abs(best_solution - whales[i])
                whales[i] = distance_to_best * np.exp(a2 * np.random.rand()) * np.cos(2 * np.pi * np.random.rand()) + best_solution
        
        for i in range(n):
            fitness[i] = objective_function(whales[i])
        
        best_index = np.argmin(fitness)
        if fitness[best_index] < best_fitness:
            best_solution = whales[best_index]
            best_fitness = fitness[best_index]
        
        print(f"Iteration {t+1}, Best fitness: {best_fitness}")

    return best_solution, best_fitness

# Parameters
n = 30              # Number of whales
dim = 10            # Number of dimensions
lb = -5.12          # Lower bound of variables
ub = 5.12           # Upper bound of variables
n_iterations = 100  # Number of iterations

# Run Whale Optimization Algorithm
best_solution, best_solution_fitness = whale_optimization_algorithm(n, dim, lb, ub, n_iterations)
print(f"Best solution: {best_solution}")
print(f"Best solution fitness: {best_solution_fitness}")
