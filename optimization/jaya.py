"""
JAYA Optimizer Implementation
Simple and parameter-free metaheuristic optimization algorithm

Reference: Rao, R. (2016). Jaya: A simple and new optimization algorithm 
for solving constrained and unconstrained optimization problems. 
International Journal of Industrial Engineering Computations, 7(1), 19-34.
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer

class JAYAOptimizer(BaseOptimizer):
    """
    JAYA Optimization Algorithm
    
    JAYA means "victory" in Sanskrit. It's a simple, parameter-free algorithm
    that moves towards the best solution and away from the worst solution.
    
    Update equation:
    X_new = X_old + r1*(X_best - |X_old|) - r2*(X_worst - |X_old|)
    """
    
    def __init__(self, 
                 objective_func,
                 bounds: Dict[str, list],
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-10,
                 verbose: bool = False):
        """
        Initialize JAYA optimizer
        
        Args:
            objective_func: Function to minimize
            bounds: Dictionary with parameter bounds {'param': [min, max]}
            population_size: Size of population
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            verbose: Print progress information
        """
        super().__init__(objective_func, bounds, population_size, 
                        max_iterations, tolerance)
        self.verbose = verbose
        
    def optimize(self) -> Tuple[Dict[str, float], float]:
        """
        Run JAYA optimization algorithm
        
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        # Update best solution
        self.update_best(population, fitness)
        self.convergence_history.append(self.best_fitness)
        
        if self.verbose:
            print(f"Initial best fitness: {self.best_fitness:.8e}")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            # Find best and worst solutions in current population
            best_idx = np.argmin(fitness)
            worst_idx = np.argmax(fitness)
            
            best_solution = population[best_idx]
            worst_solution = population[worst_idx]
            
            # Create new population using JAYA update equation
            new_population = np.zeros_like(population)
            
            for i in range(self.population_size):
                # Generate random numbers r1 and r2 for each dimension
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                
                # JAYA update equation
                # Move towards best solution and away from worst solution
                new_individual = (population[i] + 
                                r1 * (best_solution - np.abs(population[i])) -
                                r2 * (worst_solution - np.abs(population[i])))
                
                # Apply boundary constraints
                new_individual = self.bound_constraint(new_individual, population[i])
                new_population[i] = new_individual
            
            # Evaluate new population
            new_fitness = self.evaluate_population(new_population)
            
            # Selection: Keep better solutions
            for i in range(self.population_size):
                if new_fitness[i] < fitness[i]:
                    population[i] = new_population[i]
                    fitness[i] = new_fitness[i]
            
            # Update best solution
            self.update_best(population, fitness)
            self.convergence_history.append(self.best_fitness)
            
            # Print progress
            if self.verbose and (iteration + 1) % 100 == 0:
                print(f"Iteration {iteration + 1:4d}: Best fitness = {self.best_fitness:.8e}")
            
            # Check convergence
            if self.check_convergence():
                if self.verbose:
                    print(f"Converged at iteration {iteration + 1}")
                break
        
        if self.verbose:
            print(f"Final best fitness: {self.best_fitness:.8e}")
            print(f"Total iterations: {self.iteration_count}")
        
        # Return results
        results = self.get_results()
        return results['best_parameters'], self.best_fitness
    
    def get_algorithm_info(self) -> Dict[str, str]:
        """
        Get algorithm information
        
        Returns:
            Dictionary with algorithm details
        """
        return {
            'name': 'JAYA',
            'type': 'Population-based metaheuristic',
            'parameters': 'Parameter-free',
            'reference': 'Rao, R. (2016). Jaya: A simple and new optimization algorithm',
            'characteristics': [
                'Simple implementation',
                'No algorithm-specific parameters',
                'Moves towards best and away from worst',
                'Suitable for constrained and unconstrained problems'
            ]
        }