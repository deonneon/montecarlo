import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from datetime import datetime, timedelta
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class MonteCarloSimulator:
    def __init__(self, n_simulations: int = 100, holiday_factor: float = 0.1):
        self.n_simulations = n_simulations
        self.holiday_factor = holiday_factor  # Factor to reduce hours on holidays

    def is_holiday(self, date: datetime, holidays_list: List[datetime]) -> bool:
        """Check if a given date is a holiday"""
        return date in holidays_list

    def generate_scenarios(
        self,
        model,
        historical_data: pd.DataFrame,
        forecast_horizon: int,
        holidays_list: Optional[List[datetime]] = None,
        confidence_level: float = 0.95,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Generate Monte Carlo scenarios using Holt-Winters predictions with holiday adjustments
        """
        scenarios = np.zeros((self.n_simulations, forecast_horizon))
        residuals = model.resid

        # Generate forecast dates
        last_date = pd.to_datetime(historical_data["date"].max())
        forecast_dates = [
            last_date + timedelta(days=x + 1) for x in range(forecast_horizon)
        ]

        for i in range(self.n_simulations):
            # Generate base forecast
            forecast = model.forecast(forecast_horizon)

            # Generate random residuals
            random_residuals = np.random.choice(
                residuals, size=forecast_horizon, replace=True
            )

            # Adjust forecasted values
            adjusted_forecast = forecast + random_residuals

            # Apply holiday adjustments if holidays list is provided
            if holidays_list is not None:
                for j, date in enumerate(forecast_dates):
                    if self.is_holiday(date, holidays_list):
                        # Reduce hours for holidays
                        adjusted_forecast[j] *= self.holiday_factor

            # Ensure non-negative forecasts
            adjusted_forecast = np.maximum(adjusted_forecast, 0)
            scenarios[i, :] = adjusted_forecast

        # Calculate confidence interval bounds
        lower_percentile = ((1 - confidence_level) / 2) * 100
        upper_percentile = (1 - (1 - confidence_level) / 2) * 100

        mean_forecast = np.mean(scenarios, axis=0)
        lower_bound = np.percentile(scenarios, lower_percentile, axis=0)
        upper_bound = np.percentile(scenarios, upper_percentile, axis=0)

        return mean_forecast, lower_bound, upper_bound
