# src/hybrid_predictor.py

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from datetime import datetime, timedelta
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class HybridLaborPredictor:
    def __init__(self, aggregate_weight: float = 0.6):
        """
        Initialize the hybrid predictor with weights for each model

        Args:
            aggregate_weight: Weight given to aggregate Monte Carlo predictions (0-1)
                            Remaining weight (1-aggregate_weight) goes to worker-level predictions
        """
        self.aggregate_weight = aggregate_weight
        self.worker_weight = 1 - aggregate_weight
        self.aggregate_simulator = MonteCarloSimulator()
        self.worker_predictor = WorkerMonteCarloPredictor()

    def calculate_employee_growth_rate(self, df: pd.DataFrame) -> float:
        """Calculate the employee growth rate based on historical data"""
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

        monthly_growth_rate = (final_count / initial_count) ** (1 / num_months) - 1
        return monthly_growth_rate

    def generate_hybrid_forecast(
        self,
        df: pd.DataFrame,
        forecast_horizon: int,
        include_seasonality: bool = True,
        include_growth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a hybrid forecast combining aggregate and worker-level predictions

        Args:
            df: Input DataFrame with labor data
            forecast_horizon: Number of days to forecast
            include_seasonality: Whether to include seasonal adjustments
            include_growth: Whether to include employee growth projections

        Returns:
            Tuple of (mean_forecast, lower_bound, upper_bound)
        """
        # 1. Generate aggregate-level forecast
        ts_data = df.groupby("date")["total_hours_charged"].sum().values

        # Fit Holt-Winters model with appropriate seasonality
        seasonal_period = (
            252 if include_seasonality else None
        )  # 252 working days per year
        model = ExponentialSmoothing(
            ts_data,
            seasonal_periods=seasonal_period,
            trend="add",
            seasonal="add" if include_seasonality else None,
            damped_trend=True,
        ).fit(optimized=True)

        agg_mean, agg_lower, agg_upper = self.aggregate_simulator.generate_scenarios(
            model, df, forecast_horizon
        )

        # 2. Generate worker-level forecast
        worker_predictions = self.worker_predictor.predict_all_workers(
            df, forecast_horizon
        )

        # Combine worker predictions
        total_mean_prediction = np.zeros(forecast_horizon)
        total_lower_bound = np.zeros(forecast_horizon)
        total_upper_bound = np.zeros(forecast_horizon)

        for worker_id, prediction in worker_predictions.items():
            total_mean_prediction += prediction["mean_prediction"]
            total_lower_bound += prediction["lower_bound"]
            total_upper_bound += prediction["upper_bound"]

        # 3. Calculate employee growth factor if requested
        growth_factor = 1.0
        if include_growth:
            monthly_growth_rate = self.calculate_employee_growth_rate(df)
            growth_factor = np.array(
                [(1 + monthly_growth_rate) ** (i / 30) for i in range(forecast_horizon)]
            )

        # 4. Apply weights and combine predictions
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

        return hybrid_mean, hybrid_lower, hybrid_upper

    def get_model_diagnostics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate diagnostic metrics for the hybrid model

        Returns:
            Dictionary containing various model metrics
        """
        # Calculate aggregate model metrics
        agg_residuals = self.calculate_aggregate_residuals(df)

        # Calculate worker model metrics
        worker_residuals = self.calculate_worker_residuals(df)

        return {
            "aggregate_rmse": np.sqrt(np.mean(agg_residuals**2)),
            "worker_rmse": np.sqrt(np.mean(worker_residuals**2)),
            "aggregate_weight": self.aggregate_weight,
            "worker_weight": self.worker_weight,
        }

    def calculate_aggregate_residuals(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate residuals for the aggregate model"""
        daily_totals = df.groupby("date")["total_hours_charged"].sum()
        model = ExponentialSmoothing(
            daily_totals.values,
            seasonal_periods=252,
            trend="add",
            seasonal="add",
            damped_trend=True,
        ).fit(optimized=True)
        return model.resid

    def calculate_worker_residuals(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate residuals for the worker-level model"""
        all_residuals = []
        for worker_id in df["userid"].unique():
            worker_data = df[df["userid"] == worker_id]
            worker_stats = self.worker_predictor.calculate_worker_stats(df, worker_id)
            residuals = worker_data["total_hours_charged"] - worker_stats["mean_hours"]
            all_residuals.extend(residuals)
        return np.array(all_residuals)
