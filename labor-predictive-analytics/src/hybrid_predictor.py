import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from time_series_model import TimeSeriesPredictor


class HybridLaborPredictor:
    def __init__(self, n_simulations: int = 10):
        self.top_level_simulator = MonteCarloSimulator(n_simulations)
        self.worker_predictor = WorkerMonteCarloPredictor(n_simulations)
        self.ts_predictor = TimeSeriesPredictor()
        self.n_simulations = n_simulations
        self.historical_accuracy = {"hw": None, "worker": None}
        self.cached_base_forecast = None
        self.cached_params = None

    def calculate_employee_growth_rate(self, historical_data: pd.DataFrame) -> float:
        """Calculate the rate of employee growth over time"""
        daily_employees = historical_data.groupby("date")["userid"].nunique()
        smoothed_counts = daily_employees.rolling(window=30, min_periods=1).mean()

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
        """Calculate seasonal factors based on fiscal periods and day of week patterns"""
        try:
            if historical_data.empty:
                return pd.Series([1.0] * 12, index=range(1, 13))

            # Add day of week
            historical_data = historical_data.copy()
            historical_data["day_of_week"] = historical_data["date"].dt.dayofweek

            # Calculate patterns considering both fiscal period and day of week
            combined_pattern = historical_data.groupby(
                ["fiscal_period", "day_of_week"]
            )["total_hours_charged"].mean()

            # Average across days of week for each fiscal period
            fiscal_pattern = combined_pattern.groupby(level=0).mean()

            # Fill any missing periods with the mean
            fiscal_pattern = fiscal_pattern.reindex(
                range(1, 13), fill_value=fiscal_pattern.mean()
            )

            # Normalize around 1.0 while preserving relative differences
            seasonal_factors = fiscal_pattern / fiscal_pattern.mean()

            # Clip to prevent extreme adjustments
            seasonal_factors = seasonal_factors.clip(lower=0.85, upper=1.15)

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
        """Calculate weights for HW vs worker predictions based on historical accuracy."""
        # After updating historical_accuracy with actual values
        if (
            self.historical_accuracy["hw"] is not None
            and self.historical_accuracy["worker"] is not None
        ):
            hw_accuracy = self.historical_accuracy["hw"]
            worker_accuracy = self.historical_accuracy["worker"]
            total_accuracy = hw_accuracy + worker_accuracy
            if total_accuracy == 0:
                return 0.5, 0.5
            hw_weight = hw_accuracy / total_accuracy
            worker_weight = worker_accuracy / total_accuracy
        else:
            # Default weights favoring the Time Series model
            hw_weight = 0.7
            worker_weight = 0.3

        return hw_weight, worker_weight

    def generate_hybrid_forecast(
        self,
        data: pd.DataFrame,
        forecast_horizon: int,
        include_seasonality: bool = True,
        include_growth: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate hybrid forecast with adjustments for accuracy"""

        # Calculate seasonal adjustments if needed
        seasonal_adjustments = None
        if include_seasonality:
            seasonal_adjustments = self._calculate_seasonal_adjustments(
                data, forecast_horizon
            )

        # Generate base forecasts
        base_mean, base_lower, base_upper = self._generate_base_forecast(
            data, forecast_horizon, include_seasonality, seasonal_adjustments
        )

        # Adjust for growth if needed
        if include_growth:
            growth_factors = self._calculate_growth_factors(data, forecast_horizon)
            base_mean *= growth_factors
            base_lower *= growth_factors
            base_upper *= growth_factors

        return base_mean, base_lower, base_upper

    def _generate_base_forecast(
        self,
        data: pd.DataFrame,
        forecast_horizon: int,
        include_seasonality: bool,
        seasonal_adjustments: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the base forecast incorporating both methods"""

        # Time Series Predictions
        self.ts_predictor.fit_holtwinters(data["total_hours_charged"].values)
        hw_mean, hw_lower, hw_upper = self.top_level_simulator.generate_scenarios(
            self.ts_predictor.model, data, forecast_horizon
        )

        # Apply seasonality to time series predictions
        if include_seasonality and seasonal_adjustments is not None:
            hw_mean *= seasonal_adjustments
            hw_lower *= seasonal_adjustments
            hw_upper *= seasonal_adjustments

        # Worker Predictions
        worker_predictions = self.worker_predictor.predict_all_workers(data)
        if include_seasonality and seasonal_adjustments is not None:
            worker_predictions = self._apply_seasonality_to_worker_predictions(
                worker_predictions, seasonal_adjustments
            )
        worker_means, worker_lowers, worker_uppers = self._aggregate_worker_predictions(
            worker_predictions, forecast_horizon
        )

        # Combine Predictions with Weights
        hw_weight, worker_weight = self.calculate_prediction_weights(data)
        combined_mean = hw_weight * hw_mean + worker_weight * worker_means
        combined_lower = hw_weight * hw_lower + worker_weight * worker_lowers
        combined_upper = hw_weight * hw_upper + worker_weight * worker_uppers

        return combined_mean, combined_lower, combined_upper

    def _aggregate_worker_predictions(
        self, worker_predictions: Dict, forecast_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aggregate worker predictions over the forecast horizon."""
        worker_means = np.zeros(forecast_horizon)
        worker_lowers = np.zeros(forecast_horizon)
        worker_uppers = np.zeros(forecast_horizon)

        for worker_id, prediction in worker_predictions.items():
            # Extend the worker's predictions to match the forecast horizon
            mean_prediction = prediction["mean_prediction"]
            lower_bound = prediction["lower_bound"]
            upper_bound = prediction["upper_bound"]

            # Repeat or extrapolate predictions to match forecast horizon
            repeats = forecast_horizon // len(mean_prediction) + 1
            extended_mean = np.tile(mean_prediction, repeats)[:forecast_horizon]
            extended_lower = np.tile(lower_bound, repeats)[:forecast_horizon]
            extended_upper = np.tile(upper_bound, repeats)[:forecast_horizon]

            worker_means += extended_mean
            worker_lowers += extended_lower
            worker_uppers += extended_upper

        return worker_means, worker_lowers, worker_uppers

    def _apply_seasonality_to_worker_predictions(
        self, worker_predictions: Dict, seasonal_adjustments: np.ndarray
    ) -> Dict:
        """Adjust worker predictions for seasonality."""
        for worker_id, prediction in worker_predictions.items():
            prediction_length = len(prediction["mean_prediction"])
            # Extend or truncate the seasonal adjustments to match the prediction length
            seasonal_factors = seasonal_adjustments[:prediction_length]
            prediction["mean_prediction"] *= seasonal_factors
            prediction["lower_bound"] *= seasonal_factors
            prediction["upper_bound"] *= seasonal_factors
        return worker_predictions

    def _calculate_seasonal_adjustments(
        self, data: pd.DataFrame, forecast_horizon: int
    ) -> np.ndarray:
        """Calculate seasonal adjustments with smoothing."""
        seasonal_factors = self.calculate_seasonal_factors(data)

        forecast_dates = pd.date_range(
            start=data["date"].max() + timedelta(days=1),
            periods=forecast_horizon,
            freq="B",
        )
        forecast_periods = [(d.month - 10) % 12 + 1 for d in forecast_dates]
        adjustments = np.array([seasonal_factors.get(p, 1.0) for p in forecast_periods])

        # Smooth adjustments to prevent sharp transitions
        window_size = 5
        smoothed_adjustments = np.convolve(
            adjustments, np.ones(window_size) / window_size, mode="same"
        )

        return smoothed_adjustments

    def _calculate_growth_factors(
        self, data: pd.DataFrame, forecast_horizon: int
    ) -> np.ndarray:
        """Calculate growth factors with minimum growth guarantee"""
        base_growth_rate = self.calculate_employee_growth_rate(data)
        min_growth_rate = 0.001  # Minimum 0.1% monthly growth

        growth_rate = max(base_growth_rate, min_growth_rate)
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
        """Update historical accuracy metrics with smoothed error calculation"""
        actual_values = actual_data["total_hours_charged"].values

        # Calculate weighted MAPE giving more weight to recent predictions
        weights = np.linspace(0.5, 1.0, len(actual_values))

        hw_mape = np.average(
            np.abs((actual_values - hw_predictions) / actual_values), weights=weights
        )
        worker_mape = np.average(
            np.abs((actual_values - worker_predictions) / actual_values),
            weights=weights,
        )

        # Convert to accuracy scores
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
            "range_ratio": np.mean(upper / lower),
        }

        return metrics
