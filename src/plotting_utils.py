import numpy as np
import matplotlib.pyplot as plt
from result import Result

def plot_fitness_history(result: Result):
    '''Plot the fitness history of the algorithm'''
    x = range(result.n_iteration)
    y = np.median(result.fitness_history, axis=1)
    yerr = np.std(result.fitness_history, axis=1)
    plt.errorbar(x, y, yerr=yerr, fmt='-o', color='black')
    plt.xlabel("Generation")
    plt.ylabel("Mean Fitness")
    plt.grid()
    plt.show()