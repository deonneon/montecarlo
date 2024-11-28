import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timedelta


class WorkerMonteCarloPredictor:
    def __init__(self, n_simulations: int = 100, holiday_factor: float = 0.1):
        self.n_simulations = n_simulations
        self.holiday_factor = holiday_factor

    def simulate_worker_days(
        self,
        worker_stats: Dict,
        n_days: int,
        future_holidays: Optional[List[datetime]] = None,
    ) -> np.ndarray:
        """Simulate work days for a single worker with holiday consideration"""
        simulations = np.zeros((self.n_simulations, n_days))

        for sim in range(self.n_simulations):
            for day in range(n_days):
                current_date = datetime.now() + timedelta(days=day)

                if future_holidays and current_date in future_holidays:
                    # Minimal hours for holidays
                    base_hours = np.random.normal(
                        worker_stats["mean_hours"] * self.holiday_factor,
                        worker_stats["std_hours"] * self.holiday_factor,
                    )
                    overtime = 0  # No overtime on holidays
                else:
                    # Regular day simulation
                    base_hours = np.random.normal(
                        worker_stats["mean_hours"], worker_stats["std_hours"]
                    )

                    # Overtime simulation
                    if np.random.random() < worker_stats["overtime_prob"]:
                        overtime = np.random.normal(
                            worker_stats["mean_overtime"], worker_stats["std_overtime"]
                        )
                        overtime = max(0, overtime)
                    else:
                        overtime = 0

                total_hours = max(0, base_hours + overtime)
                simulations[sim, day] = total_hours

        return simulations

    def predict_worker_next_week(
        self,
        df: pd.DataFrame,
        worker_id: str,
        forecast_horizon: int = 5,
        future_holidays: Optional[List[datetime]] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate predictions considering holidays"""
        worker_stats = self.calculate_worker_stats(df, worker_id)

        # Run simulations
        simulations = self.simulate_worker_days(
            worker_stats, forecast_horizon, future_holidays
        )

        # Calculate summary statistics
        summary = {
            "mean_prediction": np.mean(simulations, axis=0),
            "lower_bound": np.percentile(simulations, 2.5, axis=0),
            "upper_bound": np.percentile(simulations, 97.5, axis=0),
            "worker_stats": worker_stats,
        }

        return simulations, summary

    def predict_all_workers(
        self,
        df: pd.DataFrame,
        forecast_horizon: int = 5,
        future_holidays: Optional[List[datetime]] = None,
    ) -> Dict[str, Dict]:
        """Generate predictions for all workers"""
        worker_predictions = {}

        for worker_id in df["userid"].unique():
            _, summary = self.predict_worker_next_week(
                df, worker_id, forecast_horizon, future_holidays
            )
            worker_predictions[worker_id] = summary

        return worker_predictions

    def calculate_worker_stats(self, df: pd.DataFrame, worker_id: str) -> Dict:
        """Calculate historical statistics for a worker"""
        worker_data = df[df["userid"] == worker_id]

        # Separate holiday and non-holiday statistics
        regular_days = worker_data[~worker_data["is_holiday"]]
        holiday_days = worker_data[worker_data["is_holiday"]]

        return {
            "mean_hours": regular_days["total_hours_charged"].mean(),
            "std_hours": regular_days["total_hours_charged"].std(),
            "mean_direct": regular_days["direct_hours"].mean(),
            "std_direct": regular_days["direct_hours"].std(),
            "overtime_prob": len(regular_days[regular_days["overtime_hours"] > 0])
            / len(regular_days),
            "mean_overtime": regular_days[regular_days["overtime_hours"] > 0][
                "overtime_hours"
            ].mean(),
            "std_overtime": regular_days[regular_days["overtime_hours"] > 0][
                "overtime_hours"
            ].std(),
            "holiday_mean_hours": (
                holiday_days["total_hours_charged"].mean()
                if not holiday_days.empty
                else 0
            ),
            "holiday_std_hours": (
                holiday_days["total_hours_charged"].std()
                if not holiday_days.empty
                else 0
            ),
            "department": worker_data["dept"].mode()[0],
        }
