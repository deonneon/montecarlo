import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Tuple
from data_processing import DataPreprocessor
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from hybrid_predictor import HybridLaborPredictor
from time_series_model import TimeSeriesPredictor


class LaborForecastBenchmark:
    def __init__(
        self, train_start: str, train_end: str, test_start: str, test_end: str
    ):
        self.train_start = pd.to_datetime(train_start)
        self.train_end = pd.to_datetime(train_end)
        self.test_start = pd.to_datetime(test_start)
        self.test_end = pd.to_datetime(test_end)

        self.preprocessor = DataPreprocessor()
        self.simulator = MonteCarloSimulator()
        self.worker_predictor = WorkerMonteCarloPredictor()
        self.hybrid_predictor = HybridLaborPredictor()
        self.ts_predictor = TimeSeriesPredictor()

    def get_holidays_in_range(self, start_date: datetime, end_date: datetime) -> list:
        """Get holidays within a date range"""
        holidays = [
            datetime(2023, 1, 1),  # New Year's Day
            datetime(2023, 1, 16),  # Martin Luther King Jr. Day
            datetime(2023, 2, 20),  # Presidents' Day
            datetime(2023, 5, 29),  # Memorial Day
            datetime(2023, 6, 19),  # Juneteenth
            datetime(2023, 7, 4),  # Independence Day
            datetime(2023, 9, 4),  # Labor Day
            datetime(2023, 10, 9),  # Columbus Day
            datetime(2023, 11, 10),  # Veterans Day (observed)
            datetime(2023, 11, 23),  # Thanksgiving Day
            datetime(2023, 12, 25),  # Christmas Day
        ]

        return [h for h in holidays if start_date <= h <= end_date]

    def load_and_split_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into training and testing sets"""
        df = self.preprocessor.load_and_prepare_data(data_path)

        train_data = df[
            (df["date"] >= self.train_start) & (df["date"] <= self.train_end)
        ]
        test_data = df[(df["date"] >= self.test_start) & (df["date"] <= self.test_end)]

        return train_data, test_data

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate error metrics with holiday-aware considerations"""
        # Ensure arrays are the same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]

        # Get holidays in the test period
        holidays = self.get_holidays_in_range(self.test_start, self.test_end)
        test_dates = pd.date_range(self.test_start, periods=min_length)

        # Separate holiday and non-holiday metrics
        holiday_mask = [date in holidays for date in test_dates]
        regular_mask = [not h for h in holiday_mask]

        metrics = {}

        # Calculate metrics for all days
        metrics.update(self._calculate_period_metrics(actual, predicted, "all"))

        # Calculate metrics for regular days
        if any(regular_mask):
            metrics.update(
                self._calculate_period_metrics(
                    actual[regular_mask], predicted[regular_mask], "regular"
                )
            )

        # Calculate metrics for holidays
        if any(holiday_mask):
            metrics.update(
                self._calculate_period_metrics(
                    actual[holiday_mask], predicted[holiday_mask], "holiday"
                )
            )

        return metrics

    def _calculate_period_metrics(
        self, actual: np.ndarray, predicted: np.ndarray, period_type: str
    ) -> Dict:
        """Calculate metrics for a specific period type"""
        # For holidays, we expect reduced but non-zero values
        if period_type == "holiday":
            # If all actual values are 0, this might be a data issue
            if np.all(actual == 0):
                print(
                    f"Warning: All actual values for {period_type} are 0. This might indicate a data preprocessing issue."
                )

        # Calculate MAPE excluding zero values
        non_zero_mask = actual != 0
        if np.any(non_zero_mask):
            mape = (
                np.mean(
                    np.abs(
                        (actual[non_zero_mask] - predicted[non_zero_mask])
                        / actual[non_zero_mask]
                    )
                )
                * 100
            )
        else:
            # For holidays, use a different metric when actual is 0
            if period_type == "holiday":
                mape = (
                    np.mean(np.abs(predicted))
                    / (np.mean(actual[~non_zero_mask]) + 1)
                    * 100
                )
            else:
                mape = np.nan

        # Calculate other metrics
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mae = np.mean(np.abs(actual - predicted))
        mean_actual = np.mean(actual)
        mean_predicted = np.mean(predicted)
        std_actual = np.std(actual)
        std_predicted = np.std(predicted)

        # Calculate scale error with better holiday handling
        if period_type == "holiday":
            if mean_actual == 0:
                # For holidays, compare to typical non-holiday values
                scale_error = (mean_predicted / (np.mean(predicted) + 1)) * 100
            else:
                scale_error = (mean_predicted - mean_actual) / (mean_actual + 1) * 100
        else:
            if mean_actual != 0:
                scale_error = (mean_predicted - mean_actual) / mean_actual * 100
            else:
                scale_error = np.nan

        return {
            f"{period_type}_MAPE": mape,
            f"{period_type}_RMSE": rmse,
            f"{period_type}_MAE": mae,
            f"{period_type}_Mean_Actual": mean_actual,
            f"{period_type}_Mean_Predicted": mean_predicted,
            f"{period_type}_Std_Actual": std_actual,
            f"{period_type}_Std_Predicted": std_predicted,
            f"{period_type}_Scale_Error": scale_error,
            f"{period_type}_Sample_Size": len(actual),
        }

    def run_benchmark(self, data_path: str) -> Dict:
        """Run complete benchmark analysis with holiday awareness"""
        # Load and split data
        train_data, test_data = self.load_and_split_data(data_path)

        # Calculate number of days in test period
        forecast_horizon = (self.test_end - self.test_start).days + 1

        # Get holidays for the test period
        # Use is_holiday flag from the data instead of generating holiday list
        test_holidays = test_data[test_data["is_holiday"]]["date"].tolist()

        # Prepare training data for time series model
        daily_train = (
            train_data.groupby("date")["total_hours_charged"].sum().reset_index()
        )
        train_ts_data = daily_train["total_hours_charged"].values

        # Fit time series model
        self.ts_predictor.fit_holtwinters(train_ts_data)

        # Generate predictions using different methods
        # 1. Aggregate Monte Carlo with holidays
        agg_mean, agg_lower, agg_upper = self.simulator.generate_scenarios(
            self.ts_predictor.model, daily_train, forecast_horizon, test_holidays
        )

        # 2. Worker-level predictions with holidays
        worker_predictions = self.worker_predictor.predict_all_workers(
            train_data, forecast_horizon, test_holidays
        )

        # Combine worker predictions
        worker_mean = np.zeros(forecast_horizon)
        for worker_id, prediction in worker_predictions.items():
            worker_mean += prediction["mean_prediction"]

        # 3. Hybrid predictions
        hybrid_mean, hybrid_lower, hybrid_upper = (
            self.hybrid_predictor.generate_hybrid_forecast(
                train_data,
                forecast_horizon,
                include_seasonality=True,
                include_growth=True,
            )
        )

        # Calculate actual values (daily totals)
        actual_daily = test_data.groupby("date")["total_hours_charged"].sum()
        actual_values = actual_daily.values

        # Calculate metrics for each method
        results = {
            "Aggregate": self.calculate_metrics(actual_values, agg_mean),
            "Worker": self.calculate_metrics(actual_values, worker_mean),
            "Hybrid": self.calculate_metrics(actual_values, hybrid_mean),
        }

        return results


def main():
    benchmark = LaborForecastBenchmark(
        train_start="2021-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2023-12-31",
    )

    results = benchmark.run_benchmark("data/generated/labor_data.csv")

    # Print results with holiday-specific metrics
    print("\nBenchmark Results:")
    print("-----------------")
    period_types = ["all", "regular", "holiday"]
    metrics_order = [
        "Sample_Size",
        "MAPE",
        "RMSE",
        "MAE",
        "Mean_Actual",
        "Mean_Predicted",
        "Std_Actual",
        "Std_Predicted",
        "Scale_Error",
    ]

    for method, metrics in results.items():
        print(f"\n{method} Model:")
        for period in period_types:
            print(f"\n{period.capitalize()} Days:")
            for metric_name in metrics_order:
                key = f"{period}_{metric_name}"
                if key in metrics:
                    value = metrics[key]
                    if metric_name in ["MAPE", "Scale_Error"]:
                        print(f"  {metric_name}: {value:.2f}%")
                    elif metric_name == "Sample_Size":
                        print(f"  Number of days: {int(value)}")
                    else:
                        print(f"  {metric_name}: {value:.2f}")


if __name__ == "__main__":
    main()
