"""
Visualization Utility
Functions for plotting I-V curves, convergence, parameter distributions, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class Visualizer:
    """
    Visualization utilities for solar module optimization results
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        Initialize visualizer
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.rcParams['figure.dpi'] = dpi
        plt.rcParams['font.size'] = 12
        
    def plot_iv_pv_curves(self, results: Dict[str, Any], 
                          solar_model, num_points: int = 200,
                          save_path: Optional[str] = None) -> None:
        """
        Plot I-V and P-V characteristic curves
        
        Args:
            results: Optimization results
            solar_model: SolarModuleModel instance
            num_points: Number of points for curve generation
            save_path: Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract parameters
        params = results['best_parameters']
        param_array = np.array([params['Is1'], params['a1'], params['a2'], 
                               params['Rs'], params['Rp']])
        
        # Generate curves
        voltages, currents = solar_model.generate_iv_curve(param_array, num_points)
        _, powers = solar_model.generate_pv_curve(param_array, num_points)
        
        # Plot I-V curve
        ax1.plot(voltages, currents, 'b-', linewidth=2, label='I-V Curve')
        ax1.axhline(y=solar_model.Isc, color='r', linestyle='--', alpha=0.7, 
                   label=f'Isc = {solar_model.Isc:.2f} A')
        ax1.axvline(x=solar_model.Voc, color='r', linestyle='--', alpha=0.7, 
                   label=f'Voc = {solar_model.Voc:.2f} V')
        ax1.plot(solar_model.Vm, solar_model.Im, 'ro', markersize=8, 
                label=f'MPP ({solar_model.Vm:.1f}V, {solar_model.Im:.2f}A)')
        
        ax1.set_xlabel('Voltage (V)')
        ax1.set_ylabel('Current (A)')
        ax1.set_title(f'I-V Characteristic - {solar_model.module_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xlim(0, solar_model.Voc * 1.05)
        ax1.set_ylim(0, solar_model.Isc * 1.05)
        
        # Plot P-V curve
        ax2.plot(voltages, powers, 'g-', linewidth=2, label='P-V Curve')
        max_power = solar_model.Vm * solar_model.Im
        max_power_idx = np.argmax(powers)
        ax2.plot(voltages[max_power_idx], powers[max_power_idx], 'ro', markersize=8,
                label=f'Max Power = {max_power:.2f} W')
        
        ax2.set_xlabel('Voltage (V)')
        ax2.set_ylabel('Power (W)')
        ax2.set_title(f'P-V Characteristic - {solar_model.module_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, solar_model.Voc * 1.05)
        ax2.set_ylim(0, max(powers) * 1.05)
        
        # Add parameter information
        param_text = (f"Is1: {params['Is1']:.2e} A\n"
                     f"a1: {params['a1']:.3f}\n"
                     f"a2: {params['a2']:.3f}\n"
                     f"Rs: {params['Rs']:.4f} Ω\n"
                     f"Rp: {params['Rp']:.2f} Ω\n"
                     f"Fitness: {results['best_fitness']:.2e}")
        
        ax2.text(0.02, 0.98, param_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"I-V/P-V curves saved to: {save_path}")
        
        plt.show()
    
    def plot_convergence(self, results: Dict[str, Any], 
                        log_scale: bool = True,
                        save_path: Optional[str] = None) -> None:
        """
        Plot convergence history
        
        Args:
            results: Optimization results
            log_scale: Use logarithmic scale for y-axis
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        convergence = results['convergence_history']
        iterations = range(1, len(convergence) + 1)
        
        ax.plot(iterations, convergence, 'b-', linewidth=2, marker='o', 
               markersize=4, markevery=max(1, len(convergence)//20))
        
        if log_scale:
            ax.set_yscale('log')
            ax.set_ylabel('Fitness (log scale)')
        else:
            ax.set_ylabel('Fitness')
            
        ax.set_xlabel('Iteration')
        ax.set_title('Convergence History')
        ax.grid(True, alpha=0.3)
        
        # Add final fitness annotation
        final_fitness = convergence[-1]
        ax.annotate(f'Final: {final_fitness:.2e}', 
                   xy=(len(convergence), final_fitness),
                   xytext=(len(convergence)*0.7, final_fitness*2),
                   arrowprops=dict(arrowstyle='->', alpha=0.7),
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Convergence plot saved to: {save_path}")
            
        plt.show()
    
    def plot_parallel_coordinates(self, multiple_results: List[Dict[str, Any]], 
                                 normalize: bool = True,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot parallel coordinates for multiple optimization runs
        
        Args:
            multiple_results: List of results from multiple runs
            normalize: Normalize parameters to [0,1] range
            save_path: Path to save figure
        """
        # Extract parameters from all runs
        param_data = []
        for i, result in enumerate(multiple_results):
            params = result['best_parameters'].copy()
            params['run_id'] = i + 1
            params['fitness'] = result['best_fitness']
            param_data.append(params)
        
        df = pd.DataFrame(param_data)
        
        # Normalize parameters if requested
        if normalize:
            param_cols = ['Is1', 'a1', 'a2', 'Rs', 'Rp']
            df_norm = df.copy()
            
            # Define normalization bounds (from config)
            bounds = {
                'Is1': [1e-12, 1e-6],
                'a1': [0.5, 2.0],
                'a2': [0.5, 2.0], 
                'Rs': [0.001, 1.0],
                'Rp': [50, 200]
            }
            
            for param in param_cols:
                min_val, max_val = bounds[param]
                df_norm[param] = (df[param] - min_val) / (max_val - min_val)
            
            df_plot = df_norm
            title = 'Parallel Coordinates (Normalized Parameters)'
            ylabel = 'Normalized Parameter Value'
        else:
            df_plot = df
            title = 'Parallel Coordinates (Original Parameters)'
            ylabel = 'Parameter Value'
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create parallel coordinates plot
        param_cols = ['Is1', 'a1', 'a2', 'Rs', 'Rp']
        
        # Color by fitness (best runs in darker colors)
        fitness_vals = df_plot['fitness'].values
        colors = plt.cm.viridis_r((fitness_vals - fitness_vals.min()) / 
                                 (fitness_vals.max() - fitness_vals.min()))
        
        for i, row in df_plot.iterrows():
            values = [row[col] for col in param_cols]
            ax.plot(range(len(param_cols)), values, 
                   color=colors[i], alpha=0.7, linewidth=1.5)
        
        ax.set_xticks(range(len(param_cols)))
        ax.set_xticklabels(['I$_{s1}$', 'a$_1$', 'a$_2$', 'R$_s$', 'R$_p$'])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis_r, 
                                  norm=plt.Normalize(vmin=fitness_vals.min(), 
                                                   vmax=fitness_vals.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Fitness Value')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Parallel coordinates plot saved to: {save_path}")
            
        plt.show()
    
    def plot_parameter_distribution(self, multiple_results: List[Dict[str, Any]],
                                  save_path: Optional[str] = None) -> None:
        """
        Plot parameter distribution from multiple runs
        
        Args:
            multiple_results: List of results from multiple runs
            save_path: Path to save figure
        """
        # Extract parameters
        param_data = []
        for result in multiple_results:
            param_data.append(result['best_parameters'])
        
        df = pd.DataFrame(param_data)
        param_names = ['Is1', 'a1', 'a2', 'Rs', 'Rp']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, param in enumerate(param_names):
            if param in df.columns:
                axes[i].hist(df[param], bins=15, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{param} Distribution')
                axes[i].set_xlabel(param)
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = df[param].mean()
                std_val = df[param].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', 
                              label=f'Mean: {mean_val:.3e}')
                axes[i].legend()
        
        # Plot fitness distribution in the last subplot
        fitness_vals = [result['best_fitness'] for result in multiple_results]
        axes[-1].hist(fitness_vals, bins=15, alpha=0.7, edgecolor='black', color='orange')
        axes[-1].set_title('Fitness Distribution')
        axes[-1].set_xlabel('Fitness Value')
        axes[-1].set_ylabel('Frequency')
        axes[-1].grid(True, alpha=0.3)
        axes[-1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Parameter distribution plot saved to: {save_path}")
            
        plt.show()
    
    def plot_3d_parameter_space(self, multiple_results: List[Dict[str, Any]],
                               params: List[str] = ['Rs', 'Rp', 'a1'],
                               save_path: Optional[str] = None) -> None:
        """
        3D scatter plot of parameter space
        
        Args:
            multiple_results: List of results from multiple runs
            params: List of 3 parameters to plot
            save_path: Path to save figure
        """
        if len(params) != 3:
            raise ValueError("Exactly 3 parameters must be specified for 3D plot")
        
        # Extract data
        param_data = []
        fitness_data = []
        
        for result in multiple_results:
            param_values = [result['best_parameters'][p] for p in params]
            param_data.append(param_values)
            fitness_data.append(result['best_fitness'])
        
        param_data = np.array(param_data)
        fitness_data = np.array(fitness_data)
        
        # Create 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color by fitness
        colors = plt.cm.viridis_r((fitness_data - fitness_data.min()) / 
                                 (fitness_data.max() - fitness_data.min()))
        
        scatter = ax.scatter(param_data[:, 0], param_data[:, 1], param_data[:, 2],
                           c=fitness_data, cmap='viridis_r', s=60, alpha=0.7)
        
        ax.set_xlabel(params[0])
        ax.set_ylabel(params[1])
        ax.set_zlabel(params[2])
        ax.set_title(f'3D Parameter Space ({params[0]}, {params[1]}, {params[2]})')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Fitness Value')
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"3D parameter space plot saved to: {save_path}")
            
        plt.show()
    
    def plot_comparison(self, results_list: List[Dict[str, Any]], 
                       labels: List[str], metric: str = 'fitness',
                       save_path: Optional[str] = None) -> None:
        """
        Compare results from different algorithms or runs
        
        Args:
            results_list: List of results to compare
            labels: Labels for each result set
            metric: Metric to compare ('fitness', 'iterations')
            save_path: Path to save figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if metric == 'fitness':
            values = [result['best_fitness'] for result in results_list]
            ylabel = 'Best Fitness'
            title = 'Fitness Comparison'
        elif metric == 'iterations':
            values = [result.get('total_iterations', 0) for result in results_list]
            ylabel = 'Total Iterations'
            title = 'Iterations Comparison'
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        bars = ax.bar(labels, values, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.2e}' if metric == 'fitness' else f'{value}',
                   ha='center', va='bottom')
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        if metric == 'fitness':
            ax.set_yscale('log')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
            
        plt.show()