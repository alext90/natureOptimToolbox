import numpy as np

from population import Population
from artifical_bee_colony import ArtificialBeeColony
from cuckoo_search import CuckooSearch
from bat_search import BatSearch
from firefly_search import FireflyAlgorithm
from whale_optimization import WhaleOptimizationAlgorithm
from grey_wolf import GrayWolfOptimizer

from example_functions import sphere, rosenbrock

if __name__ == "__main__":
    population_size = 50       
    dim_individual = 3      
    lb = -5                  
    ub = 5                   

    objective_function = rosenbrock

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )
    n_generations = 100   
    error_tol = 0.01
    limit = 100    
                 
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
    #result.plot_phenotypic_diversity()
    #result.plot_genotypic_diversity()

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )
    
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


    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )
    
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

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )

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

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )

    woa = WhaleOptimizationAlgorithm(population,
                                     n_generations,
                                     verbose=False,
                                     error_tol=error_tol
                                     )
    
    result = woa.run()
    print("\nWhale Optimization Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )

    gwo = GrayWolfOptimizer(population,
                            n_generations
                            )
    
    result = gwo.run()
    print("\nGray Wolf Optimizer Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")