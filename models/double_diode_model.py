"""
Double diode solar module model and optimization
"""

import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, List
from config.solar_modules import Q, K, T0, SOLAR_MODULES


class DoubleDiodeModel:
    """Double diode solar photovoltaic module model"""

    def __init__(self, module_type: str = "ST40", temperature: float = 25):
        """
        Initialize double diode solar module model

        Args:
            module_type: Type of solar module ('KC200GT', 'SQ85', 'ST40')
            temperature: Operating temperature in Celsius
        """
        if module_type not in SOLAR_MODULES:
            raise ValueError(f"Unknown module type: {module_type}")

        self.module_specs = SOLAR_MODULES[module_type]
        self.temperature = temperature
        self.T = temperature + T0  # Convert to Kelvin

        # Extract module parameters
        self.Voc = self.module_specs["Voc"]
        self.Isc = self.module_specs["Isc"]
        self.Vm = self.module_specs["Vm"]
        self.Im = self.module_specs["Im"]
        self.Nc = self.module_specs["Nc"]

    def objective_function(self, x: np.ndarray) -> float:
        """
        Calculate the objective function (sum of squared errors)

        Args:
            x: Array of parameters [Is1, a1, a2, Rs, Rp]

        Returns:
            error: Sum of squared errors
        """
        Is1, a1, a2, Rs, Rp = x

        # Calculate Is2
        num = (
            self.Isc
            + Rs * self.Isc / Rp
            - self.Voc / Rp
            - Is1
            * (
                np.exp(Q * self.Voc / (a1 * K * self.Nc * self.T))
                - np.exp(Q * Rs * self.Isc / (a1 * K * self.Nc * self.T))
            )
        )
        den = np.exp(Q * self.Voc / (a2 * K * self.Nc * self.T)) - np.exp(
            Q * Rs * self.Isc / (a2 * K * self.Nc * self.T)
        )
        Is2 = abs(num / den)

        # Calculate Iph
        Iph = (
            Is1 * (np.exp(Q * self.Voc / (a1 * K * self.Nc * self.T)) - 1)
            + Is2 * (np.exp(Q * self.Voc / (a2 * K * self.Nc * self.T)) - 1)
            + self.Voc / Rp
        )

        # Calculate errors at three key points
        # Error 1: At open circuit (V = Voc, I = 0)
        eps1 = (
            Is1 * (np.exp(Q * self.Voc / (a1 * K * self.Nc * self.T)) - 1)
            + Is2 * (np.exp(Q * self.Voc / (a2 * K * self.Nc * self.T)) - 1)
            + self.Voc / Rp
            - Iph
        )

        # Error 2: At short circuit (V = 0, I = Isc)
        eps2 = (
            self.Isc
            + Is1 * (np.exp(Q * Rs * self.Isc / (a1 * K * self.Nc * self.T)) - 1)
            + Is2 * (np.exp(Q * Rs * self.Isc / (a2 * K * self.Nc * self.T)) - 1)
            + Rs * self.Isc / Rp
            - Iph
        )

        # Error 3: At maximum power point (V = Vm, I = Im)
        eps3 = (
            Iph
            - Is1
            * (
                np.exp(Q * (self.Vm + Rs * self.Im) / (a1 * K * self.Nc * self.T))
                - 1
            )
            - Is2
            * (
                np.exp(Q * (self.Vm + Rs * self.Im) / (a2 * K * self.Nc * self.T))
                - 1
            )
            - (self.Vm + Rs * self.Im) / Rp
            - self.Im
        )

        # Total error (sum of squared errors)
        error = eps1**2 + eps2**2 + eps3**2

        return error

    def calculate_iv_curve(self, x: np.ndarray, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V curve for given parameters

        Args:
            x: Array of parameters [Is1, a1, a2, Rs, Rp]
            num_points: Number of points in the curve

        Returns:
            V: Voltage array
            I: Current array
        """
        Is1, a1, a2, Rs, Rp = x

        # Calculate Is2 and Iph
        num = (
            self.Isc
            + Rs * self.Isc / Rp
            - self.Voc / Rp
            - Is1
            * (
                np.exp(Q * self.Voc / (a1 * K * self.Nc * self.T))
                - np.exp(Q * Rs * self.Isc / (a1 * K * self.Nc * self.T))
            )
        )
        den = np.exp(Q * self.Voc / (a2 * K * self.Nc * self.T)) - np.exp(
            Q * Rs * self.Isc / (a2 * K * self.Nc * self.T)
        )
        Is2 = abs(num / den)

        Iph = (
            Is1 * (np.exp(Q * self.Voc / (a1 * K * self.Nc * self.T)) - 1)
            + Is2 * (np.exp(Q * self.Voc / (a2 * K * self.Nc * self.T)) - 1)
            + self.Voc / Rp
        )

        # Generate voltage points
        V = np.linspace(0, self.Voc, num_points)
        I = np.zeros_like(V)

        # Define the implicit equation for current
        def current_equation(i, v):
            return Iph - Is1 * (
                np.exp(Q * (v + Rs * i) / (a1 * K * self.Nc * self.T)) - 1
            ) - Is2 * (
                np.exp(Q * (v + Rs * i) / (a2 * K * self.Nc * self.T)) - 1
            ) - (v + Rs * i) / Rp - i

        # Solve for current at each voltage point
        for idx, v in enumerate(V):
            try:
                I[idx] = fsolve(current_equation, self.Isc / 2, args=(v,))[0]
            except:
                I[idx] = 0

        return V, I