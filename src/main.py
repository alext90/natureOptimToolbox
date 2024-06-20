import numpy as np
from artifical_bee_colony import ArtificialBeeColony

# Objective function (example: sphere function)
def objective_function(x):
    return np.sum(x**2)

if __name__ == "__main__":
    n = 25                      # Number of individuals
    dim = 2                    # Number of dimensions
    lb = -5.12                  # Lower bound of variables
    ub = 5.12                   # Upper bound of variables
    limit = 100                 # Limit for scout bee phase
    n_iterations = 100          # Number of iterations
    pa = 0.25                   # Probability of cuckoo egg detection

    # Artificial Bee Colony Algorithm
    abc = ArtificialBeeColony(n, objective_function, n_iterations, lb, ub, dim, limit)
    best_solution, best_solution_fitness = abc.run()
    print(f"Best solution: {best_solution}")
    print(f"Best solution fitness: {best_solution_fitness}")

    # Cuckoo Search Algorithm
    #best_solution, best_solution_fitness = cuckoo_search(n, dim, lb, ub, pa, n_iterations)
    #print(f"Best solution: {best_solution}")
    #print(f"Best solution fitness: {best_solution_fitness}")
