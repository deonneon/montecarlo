import pytest
import numpy as np
from src.monte_carlo import MonteCarloSimulator


def test_monte_carlo_simulator():
    simulator = MonteCarloSimulator(n_simulations=100)
    # Add test cases
