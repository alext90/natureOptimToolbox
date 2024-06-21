# Nature Optim[ization] Toolbox
A collection of nature inspired optimization algorithms.  

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
best_solution, best_solution_fitness = abc.run()
print(f"Best solution: {best_solution}")
print(f"Best solution fitness: {best_solution_fitness}")
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
- Tests