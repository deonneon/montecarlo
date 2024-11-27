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

    def load_and_split_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and split data into training and testing sets"""
        df = self.preprocessor.load_and_prepare_data(data_path)

        train_data = df[
            (df["date"] >= self.train_start) & (df["date"] <= self.train_end)
        ]
        test_data = df[(df["date"] >= self.test_start) & (df["date"] <= self.test_end)]

        return train_data, test_data

    def calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict:
        """Calculate error metrics with handling for zero values"""
        # Ensure arrays are the same length
        min_length = min(len(actual), len(predicted))
        actual = actual[:min_length]
        predicted = predicted[:min_length]

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
            mape = np.nan

        # Calculate RMSE
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        # Calculate MAE
        mae = np.mean(np.abs(actual - predicted))

        # Calculate additional metrics
        mean_actual = np.mean(actual)
        mean_predicted = np.mean(predicted)
        std_actual = np.std(actual)
        std_predicted = np.std(predicted)

        return {
            "MAPE": mape,
            "RMSE": rmse,
            "MAE": mae,
            "Mean_Actual": mean_actual,
            "Mean_Predicted": mean_predicted,
            "Std_Actual": std_actual,
            "Std_Predicted": std_predicted,
            "Scale_Error": (mean_predicted - mean_actual)
            / mean_actual
            * 100,  # Percentage error in scale
        }

    def run_benchmark(self, data_path: str) -> Dict:
        """Run complete benchmark analysis"""
        # Load and split data
        train_data, test_data = self.load_and_split_data(data_path)

        # Calculate number of days in test period
        forecast_horizon = (self.test_end - self.test_start).days + 1

        # Debug print
        print(f"\nForecast horizon: {forecast_horizon}")
        print(f"Training data shape: {train_data.shape}")
        print(f"Test data shape: {test_data.shape}")

        # Prepare training data for time series model
        daily_train = (
            train_data.groupby("date")["total_hours_charged"].sum().reset_index()
        )
        train_ts_data = daily_train["total_hours_charged"].values

        # Fit time series model
        self.ts_predictor.fit_holtwinters(train_ts_data)

        # Generate predictions using different methods
        # 1. Aggregate Monte Carlo
        agg_mean, agg_lower, agg_upper = self.simulator.generate_scenarios(
            self.ts_predictor.model, daily_train, forecast_horizon
        )

        # Debug print
        print(f"Aggregate predictions shape: {agg_mean.shape}")

        # 2. Worker-level predictions
        worker_predictions = self.worker_predictor.predict_all_workers(
            train_data, forecast_horizon
        )

        # Combine worker predictions
        worker_mean = np.zeros(forecast_horizon)
        for worker_id, prediction in worker_predictions.items():
            worker_mean += prediction["mean_prediction"]

        # Debug print
        print(f"Worker predictions shape: {worker_mean.shape}")

        # 3. Hybrid predictions
        hybrid_mean, hybrid_lower, hybrid_upper = (
            self.hybrid_predictor.generate_hybrid_forecast(train_data, forecast_horizon)
        )

        # Debug print
        print(f"Hybrid predictions shape: {hybrid_mean.shape}")

        # Calculate actual values (daily totals)

        actual_daily = test_data.groupby("date")["total_hours_charged"].sum()
        actual_values = actual_daily.values

        # Print some basic statistics about the data
        print("\nData Statistics:")
        print(
            f"Training data daily totals - Mean: {train_data.groupby('date')['total_hours_charged'].sum().mean():.2f}"
        )
        print(f"Test data daily totals - Mean: {actual_daily.mean():.2f}")
        print(f"Number of zero values in test data: {np.sum(actual_values == 0)}")
        print(f"Min value in test data: {np.min(actual_values):.2f}")
        print(f"Max value in test data: {np.max(actual_values):.2f}")

        # Calculate metrics for each method
        results = {
            "Aggregate": self.calculate_metrics(actual_values, agg_mean),
            "Worker": self.calculate_metrics(actual_values, worker_mean),
            "Hybrid": self.calculate_metrics(actual_values, hybrid_mean),
        }

        return results


def main():
    # Example usage
    benchmark = LaborForecastBenchmark(
        train_start="2021-01-01",
        train_end="2022-12-31",
        test_start="2023-01-01",
        test_end="2023-12-31",
    )

    results = benchmark.run_benchmark("data/generated/labor_data.csv")

    # Print results
    print("\nBenchmark Results:")
    print("-----------------")
    metrics_order = [
        "Mean_Actual",
        "Mean_Predicted",
        "Std_Actual",
        "Std_Predicted",
        "MAPE",
        "RMSE",
        "MAE",
        "Scale_Error",
    ]

    for method, metrics in results.items():
        print(f"\n{method} Model:")
        for metric_name in metrics_order:
            value = metrics[metric_name]
            if metric_name.startswith(("MAPE", "Scale_Error")):
                print(f"{metric_name}: {value:.2f}%")
            else:
                print(f"{metric_name}: {value:.2f}")


if __name__ == "__main__":
    main()
