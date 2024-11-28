from prophet import Prophet
import pandas as pd
import numpy as np
from typing import Tuple


class TimeSeriesPredictor:
    def __init__(
        self,
        yearly_seasonal_periods: int = 252,  # Approximately 252 working days per year
        weekly_seasonal_periods: int = 5,  # 5-day work week
    ):
        self.yearly_seasonal_periods = yearly_seasonal_periods
        self.weekly_seasonal_periods = weekly_seasonal_periods
        self.model = None

    def _create_prophet_model(self):
        """Create a new Prophet model instance with specified parameters"""
        return Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
        )

    def fit_prophet(self, data: pd.DataFrame) -> None:
        """
        Fit Prophet model with multiple seasonality
        """
        # Create new Prophet instance
        self.model = self._create_prophet_model()

        # Add holiday effects
        self.model.add_country_holidays(country_name="US")

        # Add additional seasonality for fiscal year if needed
        self.model.add_seasonality(
            name="fiscal_year", period=self.yearly_seasonal_periods, fourier_order=10
        )

        # Fit the model
        self.model.fit(data)

    def predict(self, forecast_horizon: int) -> pd.DataFrame:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=forecast_horizon, freq="D", include_history=False
        )

        # Generate forecast
        forecast = self.model.predict(future)
        return forecast

    def get_forecast_components(self, forecast_horizon: int) -> pd.DataFrame:
        """Get detailed forecast components"""
        forecast = self.predict(forecast_horizon)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "holidays"]]

    def get_fiscal_year_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze pattern by fiscal period"""
        fiscal_pattern = (
            data.groupby("fiscal_period")["total_hours_charged"]
            .agg(["mean", "std"])
            .reset_index()
        )
        return fiscal_pattern
