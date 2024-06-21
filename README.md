# Nature Optim[ization] Toolbox
A collection of nature inspired optimization algorithms. Every optimization algorithm inherits from the BaseOptimizer class using the step() function from each optimization algorithm.  

Implemented so far:  
- Artificial Bee Colony  
- Cuckoo Search  
- Bat Search
- Firefly Search
- Whale Optimization Algorithm  
- Gray Wolf Optimizer

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
from example_functions import sphere

population_size = 25       
dim_individual = 2          
lb = -5.12                  
ub = 5.12                   

error_tol = 0.01
limit = 100                 
n_generations = 100         

objective_function = sphere

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
result.plot_phenotypic_diversity()
result.plot_genotypic_diversity()
```

### ToDos:  
- Refactor all methods into one optimizers.py
- More algorithms:
    - Grey Wolf Optimizer
    - Dragonfly
    - Flower Pollination Algorithm

    - Simulated Annealing
    - Particle Swarm Obtimization
    - Genetic Algorithm