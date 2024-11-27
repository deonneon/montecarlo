import numpy as np
import pandas as pd
from typing import List, Tuple
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations

    def generate_scenarios(
        self,
        model,
        historical_data: pd.DataFrame,
        forecast_horizon: int,
        confidence_level: float = 0.95,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Generate Monte Carlo scenarios using Holt-Winters predictions
        """
        scenarios = np.zeros((self.n_simulations, forecast_horizon))
        residuals = model.resid

        for i in range(self.n_simulations):
            # Generate base forecast
            forecast = model.forecast(forecast_horizon)

            # Generate random residuals by sampling from the model's residuals
            random_residuals = np.random.choice(
                residuals, size=forecast_horizon, replace=True
            )

            # Adjust forecasted values by adding residuals
            adjusted_forecast = forecast + random_residuals

            # Ensure non-negative forecasts
            adjusted_forecast = np.maximum(adjusted_forecast, 0)
            scenarios[i, :] = adjusted_forecast

        # Calculate confidence interval bounds
        # For 95% confidence level, we want 2.5th and 97.5th percentiles
        lower_percentile = 2.5  # Fixed value for lower bound
        upper_percentile = 97.5  # Fixed value for upper bound

        mean_forecast = np.maximum(np.mean(scenarios, axis=0), 0)
        lower_bound = np.maximum(np.percentile(scenarios, lower_percentile, axis=0), 0)
        upper_bound = np.maximum(np.percentile(scenarios, upper_percentile, axis=0), 0)

        return mean_forecast, lower_bound, upper_bound
