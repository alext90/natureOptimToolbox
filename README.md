# Nature Optim[ization] Toolbox
A collection of nature inspired optimization algorithms. Every optimization algorithm inherits from the BaseOptimizer class using the step() function from each optimization algorithm.  

Implemented so far:  
- Artificial Bee Colony  
- Cuckoo Search  
- Bat Search
- Firefly Search
- Whale Optimization Algorithm  

Setup venv:  
```
make setup
```

Install requirements:  
```
make install
```

Run an example script:  
```
make run
```

Test:  
```
make test
```

## Usage

```python
import numpy as np
from population import Population
from artifical_bee_colony import ArtificialBeeColony

# Minimize sphere function
def objective_function(x):
    return np.sum(x**2)

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
plot_fitness_history(result)
```

### ToDos:  
- More algorithms:
    - Grey Wolf Optimizer
    - Dragonfly
    - Flower Pollination Algorithm

    - Simulated Annealing
    - Genetic Algorithm
    - Particle Swarm Obtimization

- Refactoring:
    - Levy Flight outside of Cuckoo (e.g. utils.py)
- Improve Readme