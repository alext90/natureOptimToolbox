import copy

from population import Population
from optimizers import (
    ArtificialBeeColony, 
    BatSearch, 
    CuckooSearch, 
    FireflyAlgorithm, 
    WhaleOptimizationAlgorithm, 
    GrayWolfOptimizer
)
from example_functions import sphere, rosenbrock

if __name__ == "__main__":
    population_size = 50       
    dim_individual = 3      
    lb = -5                  
    ub = 5                   

    objective_function = sphere

    # Generate a population
    population = Population(population_size, 
                            dim_individual, 
                            lb, 
                            ub, 
                            objective_function
                            )
    
    pop = copy.deepcopy(population)
    n_generations = 1000
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
    pop = copy.deepcopy(population)    
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
    pop = copy.deepcopy(population)
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
    pop = copy.deepcopy(population)
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
    pop = copy.deepcopy(population)
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
    pop = copy.deepcopy(population)
    gwo = GrayWolfOptimizer(population,
                            n_generations
                            )
    result = gwo.run()
    print("\nGray Wolf Optimizer Algorithm")
    print(f"Best solution: {result.best_solution}")
    print(f"Best solution fitness: {result.best_fitness:.2f}")