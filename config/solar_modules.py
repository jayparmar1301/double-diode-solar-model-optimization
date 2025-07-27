"""
Solar Module Specifications
Double-diode model parameters and specifications for different PV modules
"""

# Physical constants
Q = 1.60217646e-19  # Charge of an electron (C)
K = 1.38064852e-23  # Boltzmann's constant (J/K)
T0 = 273.15         # Absolute temperature at 0°C (K)

SOLAR_MODULES = {
    'KC200GT': {
        'name': 'Polycrystalline KC200GT',
        'Voc': 32.9,    # Open circuit voltage (V)
        'Isc': 8.21,    # Short circuit current (A)
        'Vm': 26.3,     # Voltage at maximum power (V)
        'Im': 7.61,     # Current at maximum power (A)
        'Nc': 54,       # Number of cells
        'type': 'polycrystalline'
    },
    'Shell_SQ85': {
        'name': 'Monocrystalline Shell SQ85',
        'Voc': 22.2,    # Open circuit voltage (V)
        'Isc': 5.45,    # Short circuit current (A)
        'Vm': 17.2,     # Voltage at maximum power (V)
        'Im': 4.95,     # Current at maximum power (A)
        'Nc': 36,       # Number of cells
        'type': 'monocrystalline'
    },
    'ST40': {
        'name': 'Thin film ST40',
        'Voc': 23.3,    # Open circuit voltage (V)
        'Isc': 2.68,    # Short circuit current (A)
        'Vm': 16.6,     # Voltage at maximum power (V)
        'Im': 2.41,     # Current at maximum power (A)
        'Nc': 36,       # Number of cells
        'type': 'thin_film'
    }
}

# Parameter bounds for double diode optimization
DOUBLE_DIODE_BOUNDS = {
    'Is1': [1e-12, 1e-6],   # First diode saturation current (A)
    'a1': [0.5, 2.0],       # First ideality factor
    'a2': [0.5, 2.0],       # Second ideality factor
    'Rs': [0.001, 1.0],     # Series resistance (Ω)
    'Rp': [50, 200]         # Parallel resistance (Ω)
}

# Parameter bounds for single diode optimization (for backward compatibility)
SINGLE_DIODE_BOUNDS = {
    'a': [0.5, 2.0],        # Ideality factor
    'Rs': [0.001, 1.0],     # Series resistance (Ω)
    'Rp': [50, 200]         # Parallel resistance (Ω)
}

# Physical constants
PHYSICAL_CONSTANTS = {
    'q': 1.60217646e-19,    # Electron charge (C)
    'k': 1.38064852e-23,    # Boltzmann constant (J/K)
    'T0': 273.15,           # Absolute temperature at 0°C (K)
    'T_ref': 25.0           # Reference temperature (°C)
}

# Optimization parameters
OPTIMIZATION_PARAMS = {
    'population_size': 50,      # JAYA population size
    'max_iterations': 1000,     # Maximum iterations
    'temperature': 25.0,        # Operating temperature (°C)
    'tolerance': 1e-8,          # Convergence tolerance
    'bounds': SINGLE_DIODE_BOUNDS,  # Default bounds (single diode)
    'double_diode_bounds': DOUBLE_DIODE_BOUNDS  # Double diode bounds
}

# Legacy alias for backward compatibility
PARAMETER_BOUNDS = DOUBLE_DIODE_BOUNDS