import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from datetime import datetime, timedelta


class WorkerMonteCarloPredictor:
    def __init__(self, n_simulations: int = 100):
        self.n_simulations = n_simulations

    def simulate_worker_days(self, worker_stats: Dict, n_days: int) -> np.ndarray:
        """Simulate work days for a single worker"""
        simulations = np.zeros((self.n_simulations, n_days))

        for sim in range(self.n_simulations):
            for day in range(n_days):
                # Base hours simulation
                base_hours = np.random.normal(
                    worker_stats["mean_hours"], worker_stats["std_hours"]
                )

                # Overtime simulation
                if np.random.random() < worker_stats["overtime_prob"]:
                    overtime = np.random.normal(
                        worker_stats["mean_overtime"], worker_stats["std_overtime"]
                    )
                    overtime = max(0, overtime)  # Ensure non-negative
                else:
                    overtime = 0

                total_hours = max(0, base_hours + overtime)  # Ensure non-negative
                simulations[sim, day] = total_hours

        return simulations

    def predict_worker_next_week(
        self, df: pd.DataFrame, worker_id: str, forecast_horizon: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """Generate predictions for specified number of days for a worker"""
        # Calculate worker statistics
        worker_stats = self.calculate_worker_stats(df, worker_id)

        # Run simulations
        simulations = self.simulate_worker_days(worker_stats, forecast_horizon)

        # Calculate summary statistics
        summary = {
            "mean_prediction": np.mean(simulations, axis=0),
            "lower_bound": np.percentile(simulations, 2.5, axis=0),
            "upper_bound": np.percentile(simulations, 97.5, axis=0),
            "worker_stats": worker_stats,
        }

        return simulations, summary

    def predict_all_workers(
        self, df: pd.DataFrame, forecast_horizon: int = 5
    ) -> Dict[str, Dict]:
        """Generate predictions for all workers"""
        worker_predictions = {}

        for worker_id in df["userid"].unique():
            _, summary = self.predict_worker_next_week(df, worker_id, forecast_horizon)
            worker_predictions[worker_id] = summary

        return worker_predictions

    def calculate_worker_stats(self, df: pd.DataFrame, worker_id: str) -> Dict:
        """Calculate historical statistics for a worker"""
        worker_data = df[df["userid"] == worker_id]

        return {
            "mean_hours": worker_data["total_hours_charged"].mean(),
            "std_hours": worker_data["total_hours_charged"].std(),
            "mean_direct": worker_data["direct_hours"].mean(),
            "std_direct": worker_data["direct_hours"].std(),
            "overtime_prob": len(worker_data[worker_data["overtime_hours"] > 0])
            / len(worker_data),
            "mean_overtime": worker_data[worker_data["overtime_hours"] > 0][
                "overtime_hours"
            ].mean(),
            "std_overtime": worker_data[worker_data["overtime_hours"] > 0][
                "overtime_hours"
            ].std(),
            "department": worker_data["dept"].mode()[0],
        }
