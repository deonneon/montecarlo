import pandas as pd
import numpy as np
from typing import Dict


class KPICalculator:
    def __init__(self):
        pass

    def calculate_labor_efficiency(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate labor efficiency metrics"""
        metrics = {
            "direct_labor_ratio": (
                df["direct_hours"].sum() / df["total_hours_charged"].sum()
            ),
            "overtime_ratio": (
                df["overtime_hours"].sum() / df["total_hours_charged"].sum()
            ),
            "avg_hours_per_employee": (
                df["total_hours_charged"].sum() / df["employee_count"].sum()
            ),
        }
        return metrics

    def calculate_forecast_accuracy(
        self, actual: np.array, predicted: np.array
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics"""
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))

        return {"mape": mape, "rmse": rmse}
