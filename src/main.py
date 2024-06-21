import numpy as np

from population import Population
from artifical_bee_colony import ArtificialBeeColony
from cuckoo_search import CuckooSearch
from bat_search import BatSearch
from firefly_search import FireflyAlgorithm

from plotting_utils import plot_fitness_history

# Minimize sphere function
def objective_function(x):
    return np.sum(x**2)

if __name__ == "__main__":
    population_size = 25       
    dim_individual = 2          
    lb = -5.12                  
    ub = 5.12                   

    error_tol = 0.01
    limit = 100                 
    n_generations = 100         

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )

    # Artificial Bee Colony Algorithm
    abc = ArtificialBeeColony(population, 
                              limit, 
                              n_generations,
                              error_tol=error_tol,
                              verbose=False
                              )   
    result = abc.run()
    print("Artificial Bee Colony Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")
    #plot_fitness_history(result)

    ## Cuckoo Search Algorithm
    p_discovery = 0.25
    lambda_levy = 1.5

    cs = CuckooSearch(population,
                      p_discovery,
                      lambda_levy,
                      n_generations
                      )
    result = cs.run()
    print("\nCuckoo Search Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")


    # Parameters
    f_min = 0           # Minimum frequency
    f_max = 2           # Maximum frequency
    A = 0.5             # Loudness
    r = 0.5             # Pulse rate
    alpha = 0.9         # Loudness reduction factor
    gamma = 0.9         # Pulse rate increase factor
    
    bs = BatSearch(population,
                   r,
                   A,
                   alpha,
                   gamma,
                   f_min,
                   f_max,
                   n_generations
                   )
    result = bs.run()
    print("\nBat Search Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")
    #plot_fitness_history(result)

    # Parameters
    alpha = 0.5         # Randomness strength
    beta0 = 1           # Attraction coefficient base value
    gamma = 1           # Light absorption coefficient
    ff = FireflyAlgorithm(population,
                          alpha,
                          beta0,
                          gamma,
                          n_generations
                          )
    result = ff.run()
    print("\nFirefly Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")