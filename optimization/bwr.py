"""
BWR (Best-Worst-Random) Optimizer Implementation
Population-based metaheuristic optimization algorithm

The BWR algorithm uses the best solution, worst solution, and random solution
to guide the search process, moving towards the best and away from the worst.
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer

class BWROptimizer(BaseOptimizer):
    """
    BWR (Best-Worst-Random) Optimization Algorithm
    
    The algorithm moves towards the best solution using random solutions
    and moves away from the worst solution. It includes a random reinitialization
    mechanism for enhanced exploration.
    
    Update equations:
    If r4 > 0.5:
        V'j,k,i = Vj,k,i + r1*(Vj,best,i - T*Vj,random,i) - r2*(Vj,worst,i - Vj,random,i)
    Else:
        V'j,k,i = Uj - (Uj - Lj)*r3
    
    Where T is randomly chosen from {1, 2}
    """
    
    def __init__(self, 
                 objective_func,
                 bounds: Dict[str, list],
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-10,
                 verbose: bool = False):
        """
        Initialize BWR optimizer
        
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
        Run BWR optimization algorithm
        
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
            
            # Create new population using BWR update equations
            new_population = np.zeros_like(population)
            
            for i in range(self.population_size):
                # Generate random numbers r1, r2, r3, r4 for each dimension
                r1 = np.random.random(self.dim)
                r2 = np.random.random(self.dim)
                r3 = np.random.random(self.dim)
                r4 = np.random.random(self.dim)
                
                # T factor randomly takes value 1 or 2 for each dimension
                T = np.random.choice([1, 2], size=self.dim)
                
                # Select random solution from population (excluding current individual)
                available_indices = list(range(self.population_size))
                available_indices.remove(i)
                random_idx = np.random.choice(available_indices)
                random_solution = population[random_idx]
                
                # Initialize new individual
                new_individual = np.zeros(self.dim)
                
                # Apply BWR update equations dimension by dimension
                for j in range(self.dim):
                    if r4[j] > 0.5:
                        # Equation (3): Move towards best and away from worst using random
                        new_individual[j] = (population[i][j] + 
                                           r1[j] * (best_solution[j] - T[j] * random_solution[j]) -
                                           r2[j] * (worst_solution[j] - random_solution[j]))
                    else:
                        # Equation (4): Exploration through random reinitialization
                        Uj = self.upper_bounds[j]
                        Lj = self.lower_bounds[j]
                        new_individual[j] = Uj - (Uj - Lj) * r3[j]
                
                # Apply boundary constraints
                new_individual = self.bound_constraint(new_individual, population[i])
                new_population[i] = new_individual
            
            # Evaluate new population
            new_fitness = self.evaluate_population(new_population)
            
            # Selection: Keep better solutions (greedy selection)
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
            'name': 'BWR',
            'full_name': 'Best-Worst-Random',
            'type': 'Population-based metaheuristic',
            'parameters': 'Parameter-free',
            'characteristics': [
                'Uses best solution, worst solution, and random solution',
                'Moves towards best and away from worst',
                'Random reinitialization for enhanced exploration',
                'T factor adds randomness to solution influence'
            ],
            'update_mechanism': [
                'Exploitation: V\' = V + r1*(V_best - T*V_random) - r2*(V_worst - V_random)',
                'Exploration: V\' = U - (U - L)*r3'
            ]
        }