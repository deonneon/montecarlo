# src/hybrid_predictor.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from datetime import datetime, timedelta
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from prophet import Prophet


class HybridLaborPredictor:
    def __init__(self, aggregate_weight: float = 0.6):
        self.aggregate_weight = aggregate_weight
        self.worker_weight = 1 - aggregate_weight
        self.aggregate_simulator = MonteCarloSimulator()
        self.worker_predictor = WorkerMonteCarloPredictor()

    def get_future_holidays(self, start_date: datetime, horizon: int) -> List[datetime]:
        """Generate list of future holidays"""
        # This is a simplified example - you should implement your own holiday calendar
        holidays = []
        current_date = start_date

        # Example federal holidays (simplified)
        federal_holidays = [
            "2024-01-01",  # New Year's Day
            "2024-01-15",  # Martin Luther King Jr. Day
            "2024-02-19",  # Presidents' Day
            "2024-05-27",  # Memorial Day
            "2024-06-19",  # Juneteenth
            "2024-07-04",  # Independence Day
            "2024-09-02",  # Labor Day
            "2024-10-14",  # Columbus Day
            "2024-11-11",  # Veterans Day
            "2024-11-28",  # Thanksgiving Day
            "2024-12-25",  # Christmas Day
        ]

        for i in range(horizon):
            check_date = current_date + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            if date_str in federal_holidays:
                holidays.append(check_date)

        return holidays

    def generate_hybrid_forecast(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        include_seasonality: bool = True,
        include_growth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a hybrid forecast with holiday considerations"""
        # Get future holidays
        start_date = pd.to_datetime(df["date"].max()) + timedelta(days=1)
        future_holidays = self.get_future_holidays(start_date, forecast_horizon)

        # Generate aggregate-level forecast
        # Prepare data for Prophet
        prophet_data = pd.DataFrame(
            {
                "ds": df.groupby("date")["date"].first(),
                "y": df.groupby("date")["total_hours_charged"].sum(),
            }
        ).reset_index(drop=True)

        # Initialize and fit Prophet model
        model = Prophet(
            yearly_seasonality=include_seasonality,
            weekly_seasonality=include_seasonality,
            daily_seasonality=False,
            growth="linear" if include_growth else "flat",
        )

        if include_seasonality:
            model.add_country_holidays(country_name="US")

        model.fit(prophet_data)

        # Generate forecasts
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)

        # Extract the forecast components
        agg_mean = forecast.tail(forecast_horizon)["yhat"].values
        agg_lower = forecast.tail(forecast_horizon)["yhat_lower"].values
        agg_upper = forecast.tail(forecast_horizon)["yhat_upper"].values

        # Generate worker-level forecast
        worker_predictions = self.worker_predictor.predict_all_workers(
            df, forecast_horizon, future_holidays
        )

        # Combine worker predictions
        total_mean_prediction = np.zeros(forecast_horizon)
        total_lower_bound = np.zeros(forecast_horizon)
        total_upper_bound = np.zeros(forecast_horizon)

        for worker_id, prediction in worker_predictions.items():
            total_mean_prediction += prediction["mean_prediction"]
            total_lower_bound += prediction["lower_bound"]
            total_upper_bound += prediction["upper_bound"]

        # Apply growth factor if requested
        growth_factor = 1.0
        if include_growth:
            monthly_growth_rate = self.calculate_employee_growth_rate(df)
            growth_factor = np.array(
                [(1 + monthly_growth_rate) ** (i / 30) for i in range(forecast_horizon)]
            )

        # Combine predictions with weights
        hybrid_mean = (
            self.aggregate_weight * agg_mean
            + self.worker_weight * total_mean_prediction
        ) * growth_factor

        hybrid_lower = (
            self.aggregate_weight * agg_lower + self.worker_weight * total_lower_bound
        ) * growth_factor

        hybrid_upper = (
            self.aggregate_weight * agg_upper + self.worker_weight * total_upper_bound
        ) * growth_factor

        # Final holiday adjustments for combined forecast
        for i, date in enumerate(
            pd.date_range(start=start_date, periods=forecast_horizon)
        ):
            if date in future_holidays:
                holiday_factor = 0.1  # Reduced hours on holidays
                hybrid_mean[i] *= holiday_factor
                hybrid_lower[i] *= holiday_factor
                hybrid_upper[i] *= holiday_factor

        return hybrid_mean, hybrid_lower, hybrid_upper

    def get_model_diagnostics(self, df: pd.DataFrame) -> Dict:
        """Calculate diagnostic metrics for the hybrid model"""
        # Prepare data for Prophet
        prophet_data = pd.DataFrame(
            {
                "ds": df.groupby("date")["date"].first(),
                "y": df.groupby("date")["total_hours_charged"].sum(),
            }
        ).reset_index(drop=True)

        # Fit Prophet model
        model = Prophet()
        model.fit(prophet_data)

        # Get predictions for historical period
        historical_forecast = model.predict(prophet_data)
        agg_residuals = prophet_data["y"].values - historical_forecast["yhat"].values

        # Calculate worker model metrics
        worker_residuals = self.calculate_worker_residuals(df)

        return {
            "aggregate_rmse": np.sqrt(np.mean(agg_residuals**2)),
            "worker_rmse": np.sqrt(np.mean(worker_residuals**2)),
            "aggregate_weight": self.aggregate_weight,
            "worker_weight": self.worker_weight,
        }

    def calculate_worker_residuals(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate residuals for the worker-level model"""
        all_residuals = []
        for worker_id in df["userid"].unique():
            worker_data = df[df["userid"] == worker_id]
            worker_stats = self.worker_predictor.calculate_worker_stats(df, worker_id)
            residuals = worker_data["total_hours_charged"] - worker_stats["mean_hours"]
            all_residuals.extend(residuals)
        return np.array(all_residuals)

    def calculate_employee_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate the employee growth rate based on historical data"""
        # Ensure date column is datetime
        df["date"] = pd.to_datetime(df["date"])

        # Calculate monthly unique employee counts
        monthly_counts = df.groupby(pd.Grouper(key="date", freq="ME"))[
            "userid"
        ].nunique()

        if len(monthly_counts) < 2:
            return 0.0

        initial_count = monthly_counts.iloc[0]
        final_count = monthly_counts.iloc[-1]
        num_months = len(monthly_counts) - 1

        if initial_count == 0 or num_months == 0:
            return 0.0

        # Calculate monthly growth rate
        monthly_growth_rate = (final_count / initial_count) ** (1 / num_months) - 1
        return monthly_growth_rate
