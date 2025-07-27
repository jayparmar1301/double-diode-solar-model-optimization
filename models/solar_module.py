"""
Solar Module Model - Double Diode Model
Implementation of double-diode equivalent circuit model for PV modules
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.solar_modules import SOLAR_MODULES, PHYSICAL_CONSTANTS

class SolarModuleModel:
    """
    Double-diode model for photovoltaic modules
    
    The double-diode model represents the PV cell with:
    - Photocurrent source (Iph)
    - Two diodes (with saturation currents Is1, Is2 and ideality factors a1, a2)
    - Series resistance (Rs)
    - Parallel/shunt resistance (Rp)
    
    Circuit equation: I = Iph - Is1*(exp(q*(V+I*Rs)/(a1*k*T*Nc)) - 1) 
                         - Is2*(exp(q*(V+I*Rs)/(a2*k*T*Nc)) - 1) - (V+I*Rs)/Rp
    """
    
    def __init__(self, module_name: str, temperature: float = 25.0):
        """
        Initialize solar module model
        
        Args:
            module_name: Name of the module (from SOLAR_MODULES)
            temperature: Operating temperature in Celsius
        """
        if module_name not in SOLAR_MODULES:
            raise ValueError(f"Unknown module: {module_name}. Available: {list(SOLAR_MODULES.keys())}")
            
        self.module_name = module_name
        self.module_specs = SOLAR_MODULES[module_name]
        self.temperature = temperature
        
        # Extract module specifications
        self.Voc = self.module_specs['Voc']  # Open circuit voltage
        self.Isc = self.module_specs['Isc']  # Short circuit current
        self.Vm = self.module_specs['Vm']    # Voltage at max power
        self.Im = self.module_specs['Im']    # Current at max power
        self.Nc = self.module_specs['Nc']    # Number of cells
        
        # Physical constants
        self.q = PHYSICAL_CONSTANTS['q']     # Electron charge
        self.k = PHYSICAL_CONSTANTS['k']     # Boltzmann constant
        self.T0 = PHYSICAL_CONSTANTS['T0']   # Absolute temperature at 0°C
        self.T = self.T0 + temperature       # Absolute temperature
        
    def calculate_derived_parameters(self, params: np.ndarray) -> Tuple[float, float]:
        """
        Calculate Is2 and Iph from the five main parameters
        
        Args:
            params: [Is1, a1, a2, Rs, Rp]
            
        Returns:
            Tuple of (Is2, Iph)
        """
        Is1, a1, a2, Rs, Rp = params
        
        # Calculate Is2 using boundary conditions at Voc and Isc
        numerator = (self.Isc + Rs * self.Isc / Rp - self.Voc / Rp - 
                    Is1 * (np.exp(self.q * self.Voc / (a1 * self.k * self.Nc * self.T)) - 
                           np.exp(self.q * Rs * self.Isc / (a1 * self.k * self.Nc * self.T))))
        
        denominator = (np.exp(self.q * self.Voc / (a2 * self.k * self.Nc * self.T)) - 
                      np.exp(self.q * Rs * self.Isc / (a2 * self.k * self.Nc * self.T)))
        
        Is2 = abs(numerator / denominator)
        
        # Calculate photocurrent Iph
        Iph = (Is1 * (np.exp(self.q * self.Voc / (a1 * self.k * self.Nc * self.T)) - 1) + 
               Is2 * (np.exp(self.q * self.Voc / (a2 * self.k * self.Nc * self.T)) - 1) + 
               self.Voc / Rp)
        
        return Is2, Iph
    
    def objective_function(self, params: np.ndarray) -> float:
        """
        Objective function for parameter optimization
        Minimizes squared errors at key operating points
        
        Args:
            params: [Is1, a1, a2, Rs, Rp]
            
        Returns:
            Objective function value (sum of squared errors)
        """
        try:
            Is1, a1, a2, Rs, Rp = params
            
            # Check for valid parameter values
            if (Is1 <= 0 or a1 <= 0 or a2 <= 0 or Rs <= 0 or Rp <= 0):
                return 1e10
            
            # Calculate derived parameters
            Is2, Iph = self.calculate_derived_parameters(params)
            
            # Error at open circuit (V = Voc, I = 0)
            eps1 = (Is1 * (np.exp(self.q * self.Voc / (a1 * self.k * self.Nc * self.T)) - 1) + 
                   Is2 * (np.exp(self.q * self.Voc / (a2 * self.k * self.Nc * self.T)) - 1) + 
                   self.Voc / Rp - Iph)
            
            # Error at short circuit (V = 0, I = Isc)
            eps2 = (self.Isc + Is1 * (np.exp(self.q * Rs * self.Isc / (a1 * self.k * self.Nc * self.T)) - 1) + 
                   Is2 * (np.exp(self.q * Rs * self.Isc / (a2 * self.k * self.Nc * self.T)) - 1) + 
                   Rs * self.Isc / Rp - Iph)
            
            # Error at maximum power point (V = Vm, I = Im)
            eps3 = (Iph - Is1 * (np.exp(self.q * (self.Vm + Rs * self.Im) / (a1 * self.k * self.Nc * self.T)) - 1) - 
                   Is2 * (np.exp(self.q * (self.Vm + Rs * self.Im) / (a2 * self.k * self.Nc * self.T)) - 1) - 
                   (self.Vm + Rs * self.Im) / Rp - self.Im)
            
            # Sum of squared errors
            objective = eps1**2 + eps2**2 + eps3**2
            
            return objective
            
        except (OverflowError, ZeroDivisionError, ValueError):
            return 1e10
    
    def calculate_current(self, voltage: float, params: np.ndarray) -> float:
        """
        Calculate current for given voltage using double-diode model
        Solves: I = Iph - Is1*(exp(q*(V+I*Rs)/(a1*k*T*Nc)) - 1) 
                   - Is2*(exp(q*(V+I*Rs)/(a2*k*T*Nc)) - 1) - (V+I*Rs)/Rp
        
        Args:
            voltage: Terminal voltage
            params: [Is1, a1, a2, Rs, Rp]
            
        Returns:
            Current value
        """
        Is1, a1, a2, Rs, Rp = params
        Is2, Iph = self.calculate_derived_parameters(params)
        
        def current_equation(I):
            """Implicit equation to solve for current"""
            term1 = Is1 * (np.exp(self.q * (voltage + I * Rs) / (a1 * self.k * self.Nc * self.T)) - 1)
            term2 = Is2 * (np.exp(self.q * (voltage + I * Rs) / (a2 * self.k * self.Nc * self.T)) - 1)
            term3 = (voltage + I * Rs) / Rp
            return Iph - term1 - term2 - term3 - I
        
        try:
            # Initial guess for current
            I_guess = max(0, self.Isc * (1 - voltage / self.Voc))
            current = fsolve(current_equation, I_guess, xtol=1e-10)[0]
            return max(0, current)  # Current should be non-negative
        except:
            return 0
    
    def generate_iv_curve(self, params: np.ndarray, 
                         num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate I-V characteristic curve
        
        Args:
            params: [Is1, a1, a2, Rs, Rp]
            num_points: Number of points in the curve
            
        Returns:
            Tuple of (voltage_array, current_array)
        """
        voltages = np.linspace(0, self.Voc * 1.1, num_points)
        currents = np.array([self.calculate_current(v, params) for v in voltages])
        
        # Ensure monotonic decrease
        for i in range(1, len(currents)):
            if currents[i] > currents[i-1]:
                currents[i] = currents[i-1]
        
        return voltages, currents
    
    def generate_pv_curve(self, params: np.ndarray, 
                         num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate P-V characteristic curve
        
        Args:
            params: [Is1, a1, a2, Rs, Rp]
            num_points: Number of points in the curve
            
        Returns:
            Tuple of (voltage_array, power_array)
        """
        voltages, currents = self.generate_iv_curve(params, num_points)
        powers = voltages * currents
        return voltages, powers
    
    def get_module_info(self) -> Dict:
        """
        Get module information and specifications
        
        Returns:
            Dictionary with module details
        """
        return {
            'name': self.module_name,
            'specifications': self.module_specs,
            'temperature': self.temperature,
            'model_type': 'Double-diode',
            'parameters': ['Is1', 'a1', 'a2', 'Rs', 'Rp'],
            'derived_parameters': ['Is2', 'Iph']
        }