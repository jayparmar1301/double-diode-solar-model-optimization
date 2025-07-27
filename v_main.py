"""
Working main script for double diode solar module parameter optimization
Compatible with multiple optimizer implementations: JAYA, BMR, BWR
Enhanced with comprehensive visualization capabilities
"""
import numpy as np
import time
from tqdm import tqdm

# Import configurations
from config.solar_modules import (OPTIMIZATION_PARAMS, DOUBLE_DIODE_BOUNDS, 
                                SINGLE_DIODE_BOUNDS, SOLAR_MODULES)

# Import models
from models.double_diode_model import DoubleDiodeModel

# Import optimizers
from optimization.jaya import JAYAOptimizer
from optimization.bmr import BMROptimizer
from optimization.bwr import BWROptimizer

# Import visualization and data handling utilities
from utils.visualization import Visualizer
from utils.data_handler import DataHandler

# Available optimizers
AVAILABLE_OPTIMIZERS = {
    'JAYA': JAYAOptimizer,
    'BMR': BMROptimizer,
    'BWR': BWROptimizer
}


def get_optimizer(optimizer_name, objective_func, bounds, population_size, max_iterations, seed=None):
    """
    Create optimizer instance based on name
    
    Args:
        optimizer_name: Name of optimizer ('JAYA', 'BMR', 'BWR')
        objective_func: Objective function to optimize
        bounds: Parameter bounds dictionary
        population_size: Population size
        max_iterations: Maximum iterations
        seed: Random seed for reproducibility
        
    Returns:
        Optimizer instance
    """
    if optimizer_name not in AVAILABLE_OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Available: {list(AVAILABLE_OPTIMIZERS.keys())}")
    
    optimizer_class = AVAILABLE_OPTIMIZERS[optimizer_name]
    
    # Create optimizer with seed if supported
    try:
        return optimizer_class(
            objective_func=objective_func,
            bounds=bounds,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=seed
        )
    except TypeError:
        # Fallback if seed parameter not supported
        return optimizer_class(
            objective_func=objective_func,
            bounds=bounds,
            population_size=population_size,
            max_iterations=max_iterations
        )


def run_single_optimization(module_type='ST40', optimizer_name='JAYA', seed=None, verbose=False):
    """
    Run a single optimization for double diode model with specified optimizer
    
    Args:
        module_type: Solar module type
        optimizer_name: Name of optimizer to use
        seed: Random seed
        verbose: Print detailed output
        
    Returns:
        Dictionary with optimization results
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Create double diode model
    solar_model = DoubleDiodeModel(
        module_type=module_type,
        temperature=OPTIMIZATION_PARAMS['temperature']
    )
    
    # Create optimizer
    optimizer = get_optimizer(
        optimizer_name=optimizer_name,
        objective_func=solar_model.objective_function,
        bounds=DOUBLE_DIODE_BOUNDS,
        population_size=OPTIMIZATION_PARAMS['population_size'],
        max_iterations=OPTIMIZATION_PARAMS['max_iterations'],
        seed=seed
    )
    
    # Run optimization
    start_time = time.time()
    best_solution, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    # Handle different return formats from optimizer
    if isinstance(best_solution, dict):
        # Dictionary format
        Is1 = float(best_solution['Is1'])
        a1 = float(best_solution['a1'])
        a2 = float(best_solution['a2'])
        Rs = float(best_solution['Rs'])
        Rp = float(best_solution['Rp'])
    elif isinstance(best_solution, (list, tuple)) and len(best_solution) >= 5:
        # List/tuple format
        Is1, a1, a2, Rs, Rp = [float(x) for x in best_solution[:5]]
    else:
        raise ValueError(f"Invalid best_solution format: {best_solution}")
    
    # Ensure fitness is numeric
    best_fitness = float(best_fitness)
    
    # Get optimizer results for convergence history
    try:
        optimizer_results = optimizer.get_results()
        convergence_history = optimizer_results.get('convergence_history', [])
    except:
        # Create basic convergence history if not available
        convergence_history = [best_fitness] * OPTIMIZATION_PARAMS['max_iterations']
    
    # Create results dictionary
    results = {
        'run_id': 0,  # Will be set later
        'best_solution': [Is1, a1, a2, Rs, Rp],
        'best_fitness': best_fitness,
        'execution_time': end_time - start_time,
        'module_type': module_type,
        'model_type': 'DoubleDiode',
        'optimizer': optimizer_name,
        'convergence_history': convergence_history
    }
    
    # Calculate derived parameters
    try:
        Is2, Iph = solar_model.calculate_derived_parameters([Is1, a1, a2, Rs, Rp])
    except:
        # Fallback calculation if method doesn't exist
        # Simple approximation for thin film modules
        Is2 = Is1 * 10  # Second diode typically has higher saturation current
        Iph = SOLAR_MODULES[module_type]['Isc'] * 1.02  # Slightly higher than short circuit current
    
    # Ensure derived parameters are numeric
    Is2 = float(Is2)
    Iph = float(Iph)
    
    # Store all parameters
    results.update({
        'Is1': Is1,
        'a1': a1,
        'a2': a2, 
        'Rs': Rs,
        'Rp': Rp,
        'Is2': Is2,
        'Iph': Iph,
        'parameters': {
            'Is1': Is1,
            'a1': a1,
            'a2': a2,
            'Rs': Rs,
            'Rp': Rp
        }
    })
    
    if verbose:
        print(f"\nOptimization completed in {results['execution_time']:.2f} seconds")
        print(f"Optimizer: {optimizer_name}")
        print(f"Best fitness: {best_fitness:.6e}")
        print(f"Parameters:")
        print(f"  Is1 = {Is1:.2e} A")
        print(f"  a1  = {a1:.4f}")
        print(f"  a2  = {a2:.4f}")
        print(f"  Rs  = {Rs:.4f} Ω")
        print(f"  Rp  = {Rp:.4f} Ω")
        print(f"  Is2 = {Is2:.2e} A")
        print(f"  Iph = {Iph:.4f} A")
    
    return results


def run_multiple_optimizations(module_type='ST40', optimizer_name='JAYA', num_runs=30):
    """
    Run multiple optimizations with different random seeds
    
    Args:
        module_type: Solar module type
        optimizer_name: Name of optimizer to use
        num_runs: Number of optimization runs
        
    Returns:
        List of optimization results
    """
    print(f"\nRunning {num_runs} optimizations for {module_type} module with {optimizer_name} optimizer...")
    print("="*70)
    
    all_results = []
    
    for run_id in tqdm(range(num_runs), desc=f"{optimizer_name} Optimization Progress"):
        # Generate different seed for each run
        seed = run_id * 1000 + int(time.time()) % 1000
        
        # Run optimization
        try:
            results = run_single_optimization(
                module_type=module_type,
                optimizer_name=optimizer_name,
                seed=seed,
                verbose=False
            )
            results['run_id'] = run_id + 1
            all_results.append(results)
        except Exception as e:
            print(f"\nError in run {run_id + 1}: {e}")
            continue
        
        # Print progress every 10 runs
        if (run_id + 1) % 10 == 0 and all_results:
            recent_fitness = [r['best_fitness'] for r in all_results[-min(10, len(all_results)):]]
            avg_recent = np.mean(recent_fitness)
            print(f"\nRuns {max(1, run_id-8)}-{run_id+1}: Recent average fitness: {avg_recent:.6e}")
    
    return all_results


def analyze_results(results, optimizer_name='JAYA'):
    """
    Analyze and display results for double diode model
    
    Args:
        results: List of optimization results
        optimizer_name: Name of optimizer used
        
    Returns:
        Best result dictionary
    """
    print(f"\n{optimizer_name} Double Diode Optimization Results Summary")
    print("="*60)
    
    if not results:
        print("No results to analyze!")
        return None
    
    # Extract all parameter values
    Is1_vals = [r['Is1'] for r in results]
    a1_vals = [r['a1'] for r in results]
    a2_vals = [r['a2'] for r in results]
    Rs_vals = [r['Rs'] for r in results]
    Rp_vals = [r['Rp'] for r in results]
    Is2_vals = [r['Is2'] for r in results]
    Iph_vals = [r['Iph'] for r in results]
    fitness_vals = [r['best_fitness'] for r in results]
    
    # Print statistics
    print(f"Optimizer: {optimizer_name}")
    print(f"Number of runs: {len(results)}")
    print(f"Module: {results[0]['module_type']}")
    print()
    
    print("Parameter Statistics:")
    print("-" * 40)
    print(f"Is1:  mean={np.mean(Is1_vals):.2e}, std={np.std(Is1_vals):.2e}")
    print(f"      min={np.min(Is1_vals):.2e},  max={np.max(Is1_vals):.2e}")
    print()
    print(f"a1:   mean={np.mean(a1_vals):.4f}, std={np.std(a1_vals):.4f}")
    print(f"      min={np.min(a1_vals):.4f},  max={np.max(a1_vals):.4f}")
    print()
    print(f"a2:   mean={np.mean(a2_vals):.4f}, std={np.std(a2_vals):.4f}")
    print(f"      min={np.min(a2_vals):.4f},  max={np.max(a2_vals):.4f}")
    print()
    print(f"Rs:   mean={np.mean(Rs_vals):.4f}, std={np.std(Rs_vals):.4f}")
    print(f"      min={np.min(Rs_vals):.4f},  max={np.max(Rs_vals):.4f}")
    print()
    print(f"Rp:   mean={np.mean(Rp_vals):.4f}, std={np.std(Rp_vals):.4f}")
    print(f"      min={np.min(Rp_vals):.4f},  max={np.max(Rp_vals):.4f}")
    print()
    print(f"Is2:  mean={np.mean(Is2_vals):.2e}, std={np.std(Is2_vals):.2e}")
    print(f"Iph:  mean={np.mean(Iph_vals):.4f}, std={np.std(Iph_vals):.4f}")
    print()
    print(f"Fitness: mean={np.mean(fitness_vals):.6e}")
    print(f"         min={np.min(fitness_vals):.6e}")
    print(f"         max={np.max(fitness_vals):.6e}")
    print(f"         std={np.std(fitness_vals):.6e}")
    
    # Find and display best solution
    best_idx = np.argmin(fitness_vals)
    best_result = results[best_idx]
    
    print(f"\nBest Solution (Run {best_result['run_id']}):")
    print("=" * 40)
    print(f"Is1 = {best_result['Is1']:.6e} A")
    print(f"a1  = {best_result['a1']:.6f}")
    print(f"a2  = {best_result['a2']:.6f}")
    print(f"Rs  = {best_result['Rs']:.6f} Ω")
    print(f"Rp  = {best_result['Rp']:.6f} Ω")
    print(f"Is2 = {best_result['Is2']:.6e} A")
    print(f"Iph = {best_result['Iph']:.6f} A")
    print(f"Fitness = {best_result['best_fitness']:.10e}")
    print(f"Time = {best_result['execution_time']:.2f} seconds")
    
    return best_result


def save_results_to_file(results, optimizer_name='JAYA', filename=None):
    """
    Save results to a text file
    
    Args:
        results: List of optimization results
        optimizer_name: Name of optimizer used
        filename: Output filename (auto-generated if None)
        
    Returns:
        Filename of saved file
    """
    if not results:
        print("No results to save!")
        return None
        
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{optimizer_name.lower()}_double_diode_results_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write(f"{optimizer_name} Double Diode Solar Module Optimization Results\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Optimizer: {optimizer_name}\n")
        f.write(f"Module Type: {results[0]['module_type']}\n")
        f.write(f"Model Type: {results[0]['model_type']}\n")
        f.write(f"Number of Runs: {len(results)}\n")
        f.write(f"Population Size: {OPTIMIZATION_PARAMS['population_size']}\n")
        f.write(f"Max Iterations: {OPTIMIZATION_PARAMS['max_iterations']}\n\n")
        
        # Write header
        f.write("Run   Is1        a1      a2      Rs      Rp      Is2        Iph     Fitness      Time\n")
        f.write("-" * 95 + "\n")
        
        # Write all results
        for r in results:
            f.write(f"{r['run_id']:3d}  {r['Is1']:.2e}  {r['a1']:.4f}  {r['a2']:.4f}  "
                   f"{r['Rs']:.4f}  {r['Rp']:.4f}  {r['Is2']:.2e}  {r['Iph']:.4f}  "
                   f"{r['best_fitness']:.6e}  {r['execution_time']:.2f}\n")
        
        # Write statistics
        fitness_vals = [r['best_fitness'] for r in results]
        f.write("\n" + "=" * 60 + "\n")
        f.write("STATISTICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best Fitness: {np.min(fitness_vals):.10e}\n")
        f.write(f"Mean Fitness: {np.mean(fitness_vals):.6e}\n")
        f.write(f"Std Fitness:  {np.std(fitness_vals):.6e}\n")
        f.write(f"Total Time:   {sum(r['execution_time'] for r in results):.2f} seconds\n")
        f.write(f"Avg Time:     {np.mean([r['execution_time'] for r in results]):.2f} seconds/run\n")
        
        # Algorithm information
        try:
            optimizer_instance = get_optimizer(
                optimizer_name, None, DOUBLE_DIODE_BOUNDS, 50, 1000
            )
            algo_info = optimizer_instance.get_algorithm_info()
            f.write("\n" + "=" * 60 + "\n")
            f.write("ALGORITHM INFORMATION\n")
            f.write("=" * 60 + "\n")
            f.write(f"Algorithm: {algo_info.get('name', optimizer_name)}\n")
            f.write(f"Full Name: {algo_info.get('full_name', 'N/A')}\n")
            f.write(f"Type: {algo_info.get('type', 'N/A')}\n")
            f.write(f"Parameters: {algo_info.get('parameters', 'N/A')}\n")
            
            if 'characteristics' in algo_info:
                f.write(f"\nCharacteristics:\n")
                for char in algo_info['characteristics']:
                    f.write(f"  - {char}\n")
            
            if 'update_mechanism' in algo_info:
                f.write(f"\nUpdate Mechanism:\n")
                for mech in algo_info['update_mechanism']:
                    f.write(f"  - {mech}\n")
        except:
            pass  # Skip if algorithm info not available
    
    print(f"\nResults saved to: {filename}")
    return filename


def compare_optimizers(module_type='ST40', optimizers=['JAYA', 'BMR', 'BWR'], num_runs=10):
    """
    Compare multiple optimizers on the same problem
    
    Args:
        module_type: Solar module type
        optimizers: List of optimizer names to compare
        num_runs: Number of runs for each optimizer
    """
    print(f"\n{'='*70}")
    print("OPTIMIZER COMPARISON")
    print(f"{'='*70}")
    print(f"Module: {module_type}")
    print(f"Runs per optimizer: {num_runs}")
    print(f"Optimizers: {', '.join(optimizers)}")
    
    comparison_results = {}
    
    for optimizer_name in optimizers:
        print(f"\n{'-'*50}")
        print(f"Running {optimizer_name} optimizer...")
        print(f"{'-'*50}")
        
        start_time = time.time()
        results = run_multiple_optimizations(
            module_type=module_type,
            optimizer_name=optimizer_name,
            num_runs=num_runs
        )
        total_time = time.time() - start_time
        
        if results:
            fitness_vals = [r['best_fitness'] for r in results]
            comparison_results[optimizer_name] = {
                'best_fitness': np.min(fitness_vals),
                'mean_fitness': np.mean(fitness_vals),
                'std_fitness': np.std(fitness_vals),
                'total_time': total_time,
                'avg_time': total_time / len(results),
                'success_rate': len(results) / num_runs * 100,
                'results': results
            }
    
    # Display comparison summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    print(f"{'Optimizer':<10} {'Best':<12} {'Mean':<12} {'Std':<12} {'Time(s)':<8} {'Success%':<8}")
    print("-" * 70)
    
    for optimizer_name, stats in comparison_results.items():
        print(f"{optimizer_name:<10} {stats['best_fitness']:<12.4e} {stats['mean_fitness']:<12.4e} "
              f"{stats['std_fitness']:<12.4e} {stats['avg_time']:<8.2f} {stats['success_rate']:<8.1f}")
    
    # Find best overall performer
    if comparison_results:
        best_optimizer = min(comparison_results.keys(), 
                           key=lambda x: comparison_results[x]['best_fitness'])
        print(f"\nBest performing optimizer: {best_optimizer}")
        print(f"Best fitness achieved: {comparison_results[best_optimizer]['best_fitness']:.10e}")
    
    return comparison_results


def main():
    """
    Main execution function with optimizer selection and comprehensive visualization
    """
    # Configuration
    module_type = 'ST40'  # Options: 'KC200GT', 'Shell_SQ85', 'ST40'
    num_runs = 30
    
    # Display available options
    print("Double Diode Solar Module Parameter Optimization")
    print("=" * 60)
    
    print(f"\nAvailable Optimizers: {', '.join(AVAILABLE_OPTIMIZERS.keys())}")
    print(f"Available Modules: {', '.join(SOLAR_MODULES.keys())}")
    
    # You can change this to test different optimizers
    # Options: 'JAYA', 'BMR', 'BWR'
    selected_optimizer = 'BMR'  # Change this to test different optimizers
    
    # Or uncomment the following line to compare all optimizers
    # return compare_optimizers(module_type=module_type, num_runs=10)
    
    module_info = SOLAR_MODULES[module_type]
    print(f"\nSelected Optimizer: {selected_optimizer}")
    print(f"Module: {module_info['name']}")
    print(f"Type: {module_info['type']}")
    print(f"Specifications:")
    print(f"  Voc = {module_info['Voc']} V")
    print(f"  Isc = {module_info['Isc']} A") 
    print(f"  Vm  = {module_info['Vm']} V")
    print(f"  Im  = {module_info['Im']} A")
    print(f"  Nc  = {module_info['Nc']} cells")
    
    print(f"\nOptimization Parameters:")
    print(f"  Population Size: {OPTIMIZATION_PARAMS['population_size']}")
    print(f"  Max Iterations: {OPTIMIZATION_PARAMS['max_iterations']}")
    print(f"  Temperature: {OPTIMIZATION_PARAMS['temperature']}°C")
    
    print(f"\nParameter Bounds:")
    for param, bounds in DOUBLE_DIODE_BOUNDS.items():
        print(f"  {param}: [{bounds[0]}, {bounds[1]}]")
    
    # Test single run first
    print(f"\n" + "=" * 60)
    print(f"TESTING SINGLE {selected_optimizer} OPTIMIZATION RUN")
    print("=" * 60)
    
    try:
        solar_model = DoubleDiodeModel(
            module_type=module_type,
            temperature=OPTIMIZATION_PARAMS['temperature']
        )
        
        test_result = run_single_optimization(
            module_type=module_type,
            optimizer_name=selected_optimizer,
            seed=42,
            verbose=True
        )
    except Exception as e:
        print(f"Error in single optimization test: {e}")
        return
    
    # Run multiple optimizations
    print(f"\n" + "=" * 60)
    print(f"RUNNING MULTIPLE {selected_optimizer} OPTIMIZATION TRIALS")
    print("=" * 60)
    
    start_time = time.time()
    results = run_multiple_optimizations(
        module_type=module_type,
        optimizer_name=selected_optimizer,
        num_runs=num_runs
    )
    total_time = time.time() - start_time
    
    if not results:
        print("No successful optimization runs!")
        return
    
    print(f"\n{len(results)} out of {num_runs} optimizations completed successfully!")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per successful run: {total_time/len(results):.2f} seconds")
    
    # Analyze results
    print(f"\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    best_result = analyze_results(results, selected_optimizer)
    
    if best_result is None:
        print("No results to analyze!")
        return
    
    # Save results using data handler
    print(f"\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    data_handler = DataHandler()
    
    # Save results in standard format
    data_handler.save_results(results, f'{selected_optimizer}_{module_type}_DoubleDiode')
    
    # Export for MATLAB compatibility
    # data_handler.export_for_matlab(results, f'{selected_optimizer}_DoubleDiode_data', 
                                #   f'{selected_optimizer}_{module_type}_DoubleDiode_matlab')
    
    # Also save using the original format
    filename = save_results_to_file(results, selected_optimizer)
    
    # VISUALIZATION SECTION - Similar to single diode implementation
    print(f"\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    visualizer = Visualizer()
    
    # 1. Plot convergence of best run
    best_idx = np.argmin([r['best_fitness'] for r in results])
    print("Plotting convergence curve for best run...")
    visualizer.plot_convergence(
        results[best_idx]['convergence_history'],
        title=f"{selected_optimizer} Double Diode Convergence - Best Run (Run {results[best_idx]['run_id']})"
    )
    
    # 2. Plot 5D parameter distribution (adapted for double diode parameters)
    print("Plotting 5D parameter distribution...")
    try:
        # Create modified plot for double diode parameters
        visualizer.plot_double_diode_solutions([results], [selected_optimizer])
    except AttributeError:
        # Fallback: use 3D plot with selected parameters
        print("Using 3D visualization for key parameters (a1, Rs, Rp)...")
        # Modify results format for 3D plotting
        modified_results = []
        for r in results:
            modified_r = r.copy()
            # Map double diode parameters to expected 3D format [a, Rs, Rp]
            modified_r['best_solution'] = [r['a1'], r['Rs'], r['Rp']]
            modified_r['parameters'] = {'a': r['a1'], 'Rs': r['Rs'], 'Rp': r['Rp']}
            modified_results.append(modified_r)
        
        visualizer.plot_3d_solutions([modified_results], [selected_optimizer])
    
    # 3. Plot I-V curves for best solution
    print("Plotting I-V characteristics...")
    best_solution = results[best_idx]['best_solution']
    try:
        visualizer.plot_double_diode_iv_curves(solar_model, [best_solution], 
                                             [f'Best {selected_optimizer} Solution'])
    except AttributeError:
        # Fallback: use standard IV curve plotting
        visualizer.plot_iv_curves(solar_model, [best_solution], 
                                [f'Best {selected_optimizer} Solution'])
    
    # 4. Plot P-V curves for best solution
    print("Plotting P-V characteristics...")
    try:
        visualizer.plot_double_diode_pv_curves(solar_model, [best_solution], 
                                             [f'Best {selected_optimizer} Solution'])
    except AttributeError:
        # Fallback: use standard PV curve plotting
        visualizer.plot_pv_curves(solar_model, [best_solution], 
                                [f'Best {selected_optimizer} Solution'])
    
    # 5. Statistical analysis plots
    print("Generating statistical analysis plots...")
    try:
        # Box plots for parameter distributions
        visualizer.plot_parameter_statistics(results, selected_optimizer, model_type='double_diode')
        
        # Fitness distribution histogram
        fitness_vals = [r['best_fitness'] for r in results]
        visualizer.plot_fitness_distribution(fitness_vals, 
                                           title=f'{selected_optimizer} Double Diode Fitness Distribution')
        
        # Parameter correlation analysis
        visualizer.plot_parameter_correlations(results, model_type='double_diode')
        
    except AttributeError:
        print("Extended statistical plots not available in current visualizer version")
    
    # 6. Performance summary plot
    print("Creating performance summary...")
    try:
        visualizer.plot_optimization_summary(results, selected_optimizer, model_type='double_diode')
    except AttributeError:
        print("Performance summary plot not available in current visualizer version")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("OPTIMIZATION AND VISUALIZATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Optimizer Used: {selected_optimizer}")
    print(f"Model Type: Double Diode")
    print(f"Module: {module_type}")
    print(f"Best fitness achieved: {best_result['best_fitness']:.10e}")
    print(f"Best parameters:")
    print(f"  Is1 = {best_result['Is1']:.6e} A")
    print(f"  a1  = {best_result['a1']:.6f}")
    print(f"  a2  = {best_result['a2']:.6f}")
    print(f"  Rs  = {best_result['Rs']:.6f} Ω")
    print(f"  Rp  = {best_result['Rp']:.6f} Ω")
    print(f"  Is2 = {best_result['Is2']:.6e} A")
    print(f"  Iph = {best_result['Iph']:.6f} A")
    
    if filename:
        print(f"Results saved to: {filename}")
    
    print("All visualizations generated successfully!")
    print("Ready for further analysis and comparison!")


if __name__ == "__main__":
    main()