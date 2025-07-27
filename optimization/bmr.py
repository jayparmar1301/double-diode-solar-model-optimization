"""
BMR (Best-Mean-Random) Optimizer Implementation
Population-based metaheuristic optimization algorithm

The BMR algorithm uses the best solution, mean of population, and random solution
to guide the search process with both exploitation and exploration capabilities.
"""

import numpy as np
from typing import Dict, Tuple
from .base_optimizer import BaseOptimizer

class BMROptimizer(BaseOptimizer):
    """
    BMR (Best-Mean-Random) Optimization Algorithm
    
    The algorithm moves towards the best solution using mean population values
    and random solutions for exploration. It includes a random reinitialization
    mechanism for enhanced exploration.
    
    Update equations:
    If r4 > 0.5:
        V'j,k,i = Vj,k,i + r1*(Vj,best,i - T*Vj,mean,i) + r2*(Vj,best,i - Vj,random,i)
    Else:
        V'j,k,i = Uj - (Uj - Lj)*r3
    
    Where T is randomly chosen from {1, 2}
    """
    
    def __init__(self, 
                 objective_func,
                 bounds: Dict[str, list],
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-15,  # CHANGED: Updated default from 1e-10 to 1e-15
                 verbose: bool = False):
        """
        Initialize BMR optimizer
        
        Args:
            objective_func: Function to minimize
            bounds: Dictionary with parameter bounds {'param': [min, max]}
            population_size: Size of population
            max_iterations: Maximum number of iterations
            tolerance: Fitness tolerance (stops when fitness <= tolerance)
            verbose: Print progress information
        """
        super().__init__(objective_func, bounds, population_size, 
                        max_iterations, tolerance)
        self.verbose = verbose
        
    def optimize(self) -> Tuple[Dict[str, float], float]:
        """
        Run BMR optimization algorithm
        
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
            print(f"Target tolerance: {self.tolerance:.2e}")
        
        # Main optimization loop
        for iteration in range(self.max_iterations):
            self.iteration_count = iteration + 1
            
            # Find best solution in current population
            best_idx = np.argmin(fitness)
            best_solution = population[best_idx]
            
            # Calculate mean of population for each dimension
            mean_solution = np.mean(population, axis=0)
            
            # Create new population using BMR update equations
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
                
                # Apply BMR update equations dimension by dimension
                for j in range(self.dim):
                    if r4[j] > 0.2:
                        # Equation (1): Exploitation using best, mean, and random
                        new_individual[j] = (population[i][j] + 
                                           r1[j] * (best_solution[j] - T[j] * mean_solution[j]) +
                                           r2[j] * (best_solution[j] - random_solution[j]))
                    else:
                        # Equation (2): Exploration through random reinitialization
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
            
            # Check convergence (fitness tolerance)
            if self.check_convergence():
                if self.verbose:
                    print(f"✅ Converged at iteration {iteration + 1}!")
                    print(f"   Fitness {self.best_fitness:.2e} <= tolerance {self.tolerance:.2e}")
                    efficiency = (self.max_iterations - (iteration + 1)) / self.max_iterations * 100
                    print(f"🚀 Efficiency gain: {efficiency:.1f}% ({self.max_iterations - (iteration + 1)} iterations saved)")
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
            'name': 'BMR',
            'full_name': 'Best-Mean-Random',
            'type': 'Population-based metaheuristic',
            'parameters': 'Fitness tolerance = 1e-15',
            'characteristics': [
                'Uses best solution, population mean, and random solution',
                'Balanced exploitation and exploration',
                'Random reinitialization for enhanced exploration',
                'T factor adds randomness to mean influence',
                'Early stopping when fitness <= tolerance'
            ],
            'update_mechanism': [
                'Exploitation: V\' = V + r1*(V_best - T*V_mean) + r2*(V_best - V_random)',
                'Exploration: V\' = U - (U - L)*r3',
                'Stop when fitness <= 1e-15'
            ]
        }