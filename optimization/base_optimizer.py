"""
Base Optimizer Class
Abstract base class for optimization algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Tuple, Callable, Any

class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms
    """
    
    def __init__(self, 
                 objective_func: Callable,
                 bounds: Dict[str, List[float]],
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 tolerance: float = 1e-10):
        """
        Initialize base optimizer
        
        Args:
            objective_func: Function to minimize
            bounds: Dictionary with parameter bounds {'param': [min, max]}
            population_size: Size of population
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        self.objective_func = objective_func
        self.bounds = bounds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Extract parameter names, bounds arrays
        self.param_names = list(bounds.keys())
        self.dim = len(self.param_names)
        self.lower_bounds = np.array([bounds[param][0] for param in self.param_names])
        self.upper_bounds = np.array([bounds[param][1] for param in self.param_names])
        
        # Initialize tracking variables
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_history = []
        self.iteration_count = 0
        
    def bound_constraint(self, individual: np.ndarray, 
                        population: np.ndarray = None) -> np.ndarray:
        """
        Apply boundary constraints to individual
        If violated, set to middle of previous value and bound
        
        Args:
            individual: Individual to constrain
            population: Current population (for middle point calculation)
            
        Returns:
            Constrained individual
        """
        constrained = individual.copy()
        
        # Lower bound constraint
        lower_violation = constrained < self.lower_bounds
        if population is not None and lower_violation.any():
            constrained[lower_violation] = (population[lower_violation] + 
                                          self.lower_bounds[lower_violation]) / 2
        else:
            constrained[lower_violation] = self.lower_bounds[lower_violation]
            
        # Upper bound constraint
        upper_violation = constrained > self.upper_bounds
        if population is not None and upper_violation.any():
            constrained[upper_violation] = (population[upper_violation] + 
                                          self.upper_bounds[upper_violation]) / 2
        else:
            constrained[upper_violation] = self.upper_bounds[upper_violation]
            
        return constrained
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize random population within bounds
        
        Returns:
            Initial population matrix
        """
        population = np.random.uniform(
            low=self.lower_bounds,
            high=self.upper_bounds,
            size=(self.population_size, self.dim)
        )
        return population
    
    def evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """
        Evaluate fitness for entire population
        
        Args:
            population: Population matrix
            
        Returns:
            Fitness values
        """
        fitness = np.zeros(self.population_size)
        for i, individual in enumerate(population):
            fitness[i] = self.objective_func(individual)
        return fitness
    
    def update_best(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """
        Update best solution and fitness
        
        Args:
            population: Current population
            fitness: Current fitness values
        """
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = fitness[best_idx]
            self.best_solution = population[best_idx].copy()
    
    def check_convergence(self) -> bool:
        """
        SIMPLE FITNESS CONVERGENCE: Stop when fitness <= tolerance
        
        Returns:
            True if converged (fitness <= tolerance), False otherwise
        """
        return self.best_fitness <= self.tolerance
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get optimization results
        
        Returns:
            Dictionary with results
        """
        # Convert best solution to parameter dictionary
        best_params = {param: value for param, value in 
                      zip(self.param_names, self.best_solution)}
        
        return {
            'best_parameters': best_params,
            'best_fitness': self.best_fitness,
            'convergence_history': self.convergence_history,
            'total_iterations': self.iteration_count,
            'parameter_names': self.param_names,
            'bounds': self.bounds
        }
    
    @abstractmethod
    def optimize(self) -> Tuple[Dict[str, float], float]:
        """
        Run optimization algorithm
        
        Returns:
            Tuple of (best_parameters, best_fitness)
        """
        pass