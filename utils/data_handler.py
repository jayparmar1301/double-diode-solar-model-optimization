"""
Data Handler Utility
Functions for saving and loading optimization results in various formats
"""

import json
import csv
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import os
from typing import Dict, List, Any, Optional, Union
import scipy.io as sio

class DataHandler:
    """
    Handle saving and loading of optimization results and analysis data
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize data handler
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
    def generate_filename(self, algorithm: str, module: str, 
                         extension: str, timestamp: Optional[str] = None) -> str:
        """
        Generate standardized filename
        
        Args:
            algorithm: Algorithm name (e.g., 'JAYA')
            module: Module name (e.g., 'ST40')
            extension: File extension without dot
            timestamp: Custom timestamp string
            
        Returns:
            Generated filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{algorithm}_{module}_{timestamp}.{extension}"
    
    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None,
                    algorithm: str = "JAYA", module: str = "Unknown",
                    formats: List[str] = ["json"]) -> Dict[str, str]:
        """
        Save optimization results in multiple formats
        
        Args:
            results: Results dictionary from optimization
            filename: Custom filename (without extension)
            algorithm: Algorithm name for auto-generated filename
            module: Module name for auto-generated filename
            formats: List of formats ['json', 'csv', 'pickle', 'matlab']
            
        Returns:
            Dictionary mapping format to saved filename
        """
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for fmt in formats:
            if filename:
                fname = f"{filename}.{fmt}"
            else:
                fname = self.generate_filename(algorithm, module, fmt, timestamp)
            
            filepath = self.results_dir / fname
            
            try:
                if fmt == "json":
                    self._save_json(results, filepath)
                elif fmt == "csv":
                    self._save_csv(results, filepath)
                elif fmt == "pickle":
                    self._save_pickle(results, filepath)
                elif fmt == "matlab":
                    self._save_matlab(results, filepath)
                else:
                    print(f"Warning: Unknown format '{fmt}' skipped")
                    continue
                    
                saved_files[fmt] = str(filepath)
                print(f"Results saved to: {filepath}")
                
            except Exception as e:
                print(f"Error saving {fmt} format: {str(e)}")
                
        return saved_files
    
    def _save_json(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results as JSON"""
        # Convert numpy arrays to lists for JSON serialization
        json_results = self._convert_for_json(results)
        
        with open(filepath, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
    
    def _save_csv(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results as CSV"""
        # Create DataFrame with parameters and fitness
        if 'best_parameters' in results and 'best_fitness' in results:
            data = results['best_parameters'].copy()
            data['fitness'] = results['best_fitness']
            data['iterations'] = results.get('total_iterations', 0)
            
            df = pd.DataFrame([data])
            df.to_csv(filepath, index=False)
        else:
            # Save convergence history if available
            if 'convergence_history' in results:
                df = pd.DataFrame({
                    'iteration': range(len(results['convergence_history'])),
                    'fitness': results['convergence_history']
                })
                df.to_csv(filepath, index=False)
    
    def _save_pickle(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results as pickle"""
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def _save_matlab(self, results: Dict[str, Any], filepath: Path) -> None:
        """Save results as MATLAB .mat file"""
        # Convert results for MATLAB compatibility
        matlab_dict = {}
        
        for key, value in results.items():
            if isinstance(value, dict):
                # Convert nested dict to struct-like format
                for sub_key, sub_value in value.items():
                    matlab_dict[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, list):
                matlab_dict[key] = np.array(value)
            else:
                matlab_dict[key] = value
        
        sio.savemat(filepath, matlab_dict)
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy arrays and other objects for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        else:
            return obj
    
    def load_results(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load optimization results from file
        
        Args:
            filepath: Path to results file
            
        Returns:
            Results dictionary
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Results file not found: {filepath}")
        
        extension = filepath.suffix.lower()
        
        if extension == '.json':
            return self._load_json(filepath)
        elif extension == '.csv':
            return self._load_csv(filepath)
        elif extension == '.pkl' or extension == '.pickle':
            return self._load_pickle(filepath)
        elif extension == '.mat':
            return self._load_matlab(filepath)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_json(self, filepath: Path) -> Dict[str, Any]:
        """Load results from JSON"""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def _load_csv(self, filepath: Path) -> Dict[str, Any]:
        """Load results from CSV"""
        df = pd.read_csv(filepath)
        return df.to_dict('records')[0] if len(df) == 1 else df.to_dict('list')
    
    def _load_pickle(self, filepath: Path) -> Dict[str, Any]:
        """Load results from pickle"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _load_matlab(self, filepath: Path) -> Dict[str, Any]:
        """Load results from MATLAB .mat file"""
        return sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    
    def save_multiple_runs(self, runs_results: List[Dict[str, Any]], 
                          algorithm: str = "JAYA", module: str = "Unknown",
                          summary_stats: bool = True) -> str:
        """
        Save results from multiple optimization runs
        
        Args:
            runs_results: List of results from multiple runs
            algorithm: Algorithm name
            module: Module name
            summary_stats: Whether to compute summary statistics
            
        Returns:
            Path to saved summary file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Collect all results
        all_params = []
        all_fitness = []
        all_iterations = []
        
        for i, result in enumerate(runs_results):
            params = result['best_parameters']
            params['run_id'] = i + 1
            params['fitness'] = result['best_fitness']
            params['iterations'] = result.get('total_iterations', 0)
            
            all_params.append(params)
            all_fitness.append(result['best_fitness'])
            all_iterations.append(result.get('total_iterations', 0))
        
        # Create DataFrame
        df = pd.DataFrame(all_params)
        
        # Save detailed results
        detailed_filename = f"{algorithm}_{module}_multiple_runs_{timestamp}.csv"
        detailed_path = self.results_dir / detailed_filename
        df.to_csv(detailed_path, index=False)
        
        # Compute and save summary statistics
        if summary_stats:
            summary = {
                'algorithm': algorithm,
                'module': module,
                'num_runs': len(runs_results),
                'fitness_stats': {
                    'best': float(np.min(all_fitness)),
                    'worst': float(np.max(all_fitness)),
                    'mean': float(np.mean(all_fitness)),
                    'std': float(np.std(all_fitness)),
                    'median': float(np.median(all_fitness))
                },
                'iteration_stats': {
                    'min': int(np.min(all_iterations)),
                    'max': int(np.max(all_iterations)),
                    'mean': float(np.mean(all_iterations)),
                    'std': float(np.std(all_iterations))
                },
                'timestamp': timestamp
            }
            
            summary_filename = f"{algorithm}_{module}_summary_{timestamp}.json"
            summary_path = self.results_dir / summary_filename
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"Summary saved to: {summary_path}")
        
        print(f"Multiple runs results saved to: {detailed_path}")
        return str(detailed_path)
    
    def list_results(self, algorithm: Optional[str] = None, 
                    module: Optional[str] = None) -> List[str]:
        """
        List saved results files
        
        Args:
            algorithm: Filter by algorithm name
            module: Filter by module name
            
        Returns:
            List of matching filenames
        """
        pattern = "*"
        if algorithm:
            pattern = f"{algorithm}_*"
        if module:
            if algorithm:
                pattern = f"{algorithm}_{module}_*"
            else:
                pattern = f"*_{module}_*"
        
        return [f.name for f in self.results_dir.glob(pattern)]