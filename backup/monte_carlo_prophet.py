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
        Generate Monte Carlo scenarios using Prophet predictions with holiday adjustments
        """
        scenarios = np.zeros((self.n_simulations, forecast_horizon))

        # Prepare data for Prophet
        prophet_data = pd.DataFrame(
            {"ds": historical_data["date"], "y": historical_data["total_hours_charged"]}
        )

        # Get Prophet forecast
        forecast = model.predict(model.make_future_dataframe(periods=forecast_horizon))
        forecast_dates = forecast.tail(forecast_horizon)["ds"]

        # Calculate residuals from historical predictions
        historical_forecast = model.predict(prophet_data)
        residuals = (
            historical_data["total_hours_charged"].values
            - historical_forecast["yhat"].values
        )

        for i in range(self.n_simulations):
            # Get base forecast
            base_forecast = forecast.tail(forecast_horizon)["yhat"].values

            # Generate random residuals
            random_residuals = np.random.choice(
                residuals, size=forecast_horizon, replace=True
            )

            # Adjust forecasted values
            adjusted_forecast = base_forecast + random_residuals

            # Apply holiday adjustments
            if holidays_list is not None:
                for j, date in enumerate(forecast_dates):
                    if self.is_holiday(date, holidays_list):
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
