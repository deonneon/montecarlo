import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from time_series_model import TimeSeriesPredictor


class HybridLaborPredictor:
    def __init__(self, n_simulations: int = 1000):
        self.top_level_simulator = MonteCarloSimulator(n_simulations)
        self.worker_predictor = WorkerMonteCarloPredictor(n_simulations)
        self.ts_predictor = TimeSeriesPredictor()
        self.n_simulations = n_simulations
        self.historical_accuracy = {"hw": None, "worker": None}
        # Add cache for base forecasts
        self.cached_base_forecast = None
        self.cached_params = None

    def calculate_employee_growth_rate(self, historical_data: pd.DataFrame) -> float:
        """Calculate the rate of employee growth over time"""
        daily_employees = historical_data.groupby("date")["userid"].nunique()

        # Calculate rolling 30-day average to smooth fluctuations
        smoothed_counts = daily_employees.rolling(window=30, min_periods=1).mean()

        # Calculate average monthly growth rate
        start_count = smoothed_counts.iloc[0]
        end_count = smoothed_counts.iloc[-1]
        n_months = (
            historical_data["date"].max() - historical_data["date"].min()
        ).days / 30

        if n_months == 0 or start_count == 0:
            return 0

        monthly_growth_rate = (end_count / start_count) ** (1 / n_months) - 1
        return monthly_growth_rate

    def calculate_seasonal_factors(self, historical_data: pd.DataFrame) -> pd.Series:
        """Calculate seasonal factors based on fiscal periods"""
        try:
            if historical_data.empty:
                return pd.Series([1.0] * 12, index=range(1, 13))

            fiscal_pattern = historical_data.groupby("fiscal_period")[
                "total_hours_charged"
            ].mean()
            # Fill any missing periods with 1.0
            fiscal_pattern = fiscal_pattern.reindex(range(1, 13), fill_value=1.0)
            seasonal_factors = fiscal_pattern / fiscal_pattern.mean()
            return seasonal_factors
        except Exception as e:
            print(f"Error calculating seasonal factors: {str(e)}")
            return pd.Series([1.0] * 12, index=range(1, 13))

    def calculate_department_weights(
        self, historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate weights for each department based on historical contribution"""
        dept_totals = historical_data.groupby("dept")["total_hours_charged"].sum()
        total_hours = dept_totals.sum()
        dept_weights = (dept_totals / total_hours).to_dict()
        return dept_weights

    def calculate_prediction_weights(
        self, historical_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate weights for HW vs worker predictions based on historical accuracy"""
        if (
            self.historical_accuracy["hw"] is None
            or self.historical_accuracy["worker"] is None
        ):
            # Default weights if no historical accuracy data
            return 0.6, 0.4

        hw_accuracy = self.historical_accuracy["hw"]
        worker_accuracy = self.historical_accuracy["worker"]

        total_accuracy = hw_accuracy + worker_accuracy
        if total_accuracy == 0:
            return 0.5, 0.5

        hw_weight = hw_accuracy / total_accuracy
        worker_weight = worker_accuracy / total_accuracy

        return hw_weight, worker_weight

    def generate_hybrid_forecast(
        self,
        data: pd.DataFrame,
        forecast_horizon: int,
        include_seasonality: bool = True,
        include_growth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate hybrid forecast with caching for better performance"""

        # Check if we can use cached forecast
        cache_key = (data.shape, forecast_horizon, include_seasonality)
        if self.cached_base_forecast is None or self.cached_params != cache_key:
            # Generate base forecast without growth
            base_mean, base_lower, base_upper = self._generate_base_forecast(
                data, forecast_horizon, include_seasonality
            )
            self.cached_base_forecast = (base_mean, base_lower, base_upper)
            self.cached_params = cache_key
        else:
            base_mean, base_lower, base_upper = self.cached_base_forecast

        # Apply growth if requested
        if include_growth:
            growth_factors = self._calculate_growth_factors(data, forecast_horizon)
            return (
                base_mean * growth_factors,
                base_lower * growth_factors,
                base_upper * growth_factors,
            )

        return base_mean, base_lower, base_upper

    def _generate_base_forecast(
        self, data: pd.DataFrame, forecast_horizon: int, include_seasonality: bool
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the base forecast without growth factors"""
        # Get top-level Holt-Winters predictions
        self.ts_predictor.fit_holtwinters(data["total_hours_charged"].values)
        hw_mean, hw_lower, hw_upper = self.top_level_simulator.generate_scenarios(
            self.ts_predictor.model, data, forecast_horizon
        )

        # Get worker-level predictions and aggregate them
        worker_predictions = self.worker_predictor.predict_all_workers(data)
        worker_means, worker_lowers, worker_uppers = self._aggregate_worker_predictions(
            worker_predictions, forecast_horizon
        )

        # Get weights for combining predictions
        hw_weight, worker_weight = self.calculate_prediction_weights(data)

        # Combine predictions
        combined_mean = hw_weight * hw_mean + worker_weight * worker_means
        combined_lower = hw_weight * hw_lower + worker_weight * worker_lowers
        combined_upper = hw_weight * hw_upper + worker_weight * worker_uppers

        # Apply seasonal adjustments if requested
        if include_seasonality:
            seasonal_adjustments = self._calculate_seasonal_adjustments(
                data, forecast_horizon
            )
            combined_mean *= seasonal_adjustments
            combined_lower *= seasonal_adjustments
            combined_upper *= seasonal_adjustments

        return combined_mean, combined_lower, combined_upper

    def _aggregate_worker_predictions(
        self, worker_predictions: Dict, forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate worker predictions"""
        worker_means = np.zeros(forecast_horizon)
        worker_lowers = np.zeros(forecast_horizon)
        worker_uppers = np.zeros(forecast_horizon)

        worker_forecast_days = 5
        for worker_id, prediction in worker_predictions.items():
            num_full_weeks = forecast_horizon // worker_forecast_days
            remainder_days = forecast_horizon % worker_forecast_days

            for week in range(num_full_weeks):
                start_idx = week * worker_forecast_days
                end_idx = start_idx + worker_forecast_days
                worker_means[start_idx:end_idx] += prediction["mean_prediction"]
                worker_lowers[start_idx:end_idx] += prediction["lower_bound"]
                worker_uppers[start_idx:end_idx] += prediction["upper_bound"]

            if remainder_days > 0:
                start_idx = num_full_weeks * worker_forecast_days
                worker_means[start_idx:] += prediction["mean_prediction"][
                    :remainder_days
                ]
                worker_lowers[start_idx:] += prediction["lower_bound"][:remainder_days]
                worker_uppers[start_idx:] += prediction["upper_bound"][:remainder_days]

        return worker_means, worker_lowers, worker_uppers

    def _calculate_seasonal_adjustments(
        self, data: pd.DataFrame, forecast_horizon: int
    ) -> np.ndarray:
        """Calculate seasonal adjustments"""
        seasonal_factors = self.calculate_seasonal_factors(data)
        forecast_periods = pd.date_range(
            start=data["date"].max() + timedelta(days=1),
            periods=forecast_horizon,
            freq="B",
        )
        forecast_fiscal_periods = [(d.month - 10) % 12 + 1 for d in forecast_periods]
        return np.array([seasonal_factors[p] for p in forecast_fiscal_periods])

    def _calculate_growth_factors(
        self, data: pd.DataFrame, forecast_horizon: int
    ) -> np.ndarray:
        """Calculate growth factors"""
        growth_rate = self.calculate_employee_growth_rate(data)
        return np.array(
            [(1 + growth_rate) ** (i / 30) for i in range(forecast_horizon)]
        )

    def generate_department_forecasts(
        self, data: pd.DataFrame, forecast_horizon: int
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Generate separate forecasts for each department"""
        dept_forecasts = {}

        for dept in data["dept"].unique():
            dept_data = data[data["dept"] == dept].copy()
            if len(dept_data) > 0:
                mean, lower, upper = self.generate_hybrid_forecast(
                    dept_data, forecast_horizon
                )
                dept_forecasts[dept] = (mean, lower, upper)

        return dept_forecasts

    def update_historical_accuracy(
        self,
        actual_data: pd.DataFrame,
        hw_predictions: np.ndarray,
        worker_predictions: np.ndarray,
    ) -> None:
        """Update historical accuracy metrics for both methods"""
        actual_values = actual_data["total_hours_charged"].values

        # Calculate MAPE for both methods
        hw_mape = np.mean(np.abs((actual_values - hw_predictions) / actual_values))
        worker_mape = np.mean(
            np.abs((actual_values - worker_predictions) / actual_values)
        )

        # Convert to accuracy (1 - MAPE)
        self.historical_accuracy["hw"] = 1 - hw_mape
        self.historical_accuracy["worker"] = 1 - worker_mape

    def generate_confidence_metrics(
        self, predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]
    ) -> Dict[str, float]:
        """Generate confidence metrics for the predictions"""
        mean, lower, upper = predictions

        metrics = {
            "average_uncertainty": np.mean((upper - lower) / mean),
            "max_uncertainty": np.max((upper - lower) / mean),
            "uncertainty_trend": np.polyfit(
                range(len(mean)), (upper - lower) / mean, 1
            )[0],
        }

        return metrics
