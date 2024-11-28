from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np
from typing import Tuple


class TimeSeriesPredictor:
    def __init__(
        self,
        yearly_seasonal_periods: int = 252,  # Approximately 252 working days per year
        weekly_seasonal_periods: int = 5,
    ):  # 5-day work week
        self.yearly_seasonal_periods = yearly_seasonal_periods
        self.weekly_seasonal_periods = weekly_seasonal_periods
        self.model = None

    def fit_holtwinters(self, data: np.array) -> None:
        """
        Fit Holt-Winters model with multiple seasonality
        Using yearly seasonality as primary pattern
        """
        self.model = ExponentialSmoothing(
            data,
            seasonal_periods=self.yearly_seasonal_periods,
            trend="add",
            seasonal="add",
            damped_trend=True,  # Often useful for manufacturing data
        ).fit(optimized=True)

    def predict(self, forecast_horizon: int) -> np.array:
        """Generate predictions"""
        if self.model is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.forecast(forecast_horizon)

    def get_fiscal_year_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        """Analyze pattern by fiscal period"""
        fiscal_pattern = (
            data.groupby("fiscal_period")["total_hours_charged"]
            .agg(["mean", "std"])
            .reset_index()
        )
        return fiscal_pattern
