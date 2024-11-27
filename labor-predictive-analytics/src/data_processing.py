import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessor:
    def __init__(self, fiscal_start_month: int = 10):  # October = 10
        self.fiscal_start_month = fiscal_start_month

    def load_and_prepare_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare data for time series analysis"""
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])

        # Add fiscal year and period columns
        df["fiscal_year"] = df.apply(
            lambda x: (
                x["date"].year
                if x["date"].month >= self.fiscal_start_month
                else x["date"].year - 1
            ),
            axis=1,
        )

        # Calculate fiscal period (1-12, starting from October)
        df["fiscal_period"] = df["date"].apply(
            lambda x: (x.month - self.fiscal_start_month) % 12 + 1
        )

        return df

    def aggregate_daily_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to daily level with manufacturing-specific metrics"""
        # First get the department-level aggregation
        dept_daily_metrics = (
            df.groupby(["date", "dept"])
            .agg(
                {
                    "total_hours_charged": "sum",
                    "direct_hours": "sum",
                    "overtime_hours": "sum",
                    "userid": "nunique",  # count unique employees
                }
            )
            .reset_index()
        )

        # Then get the overall daily metrics
        daily_metrics = (
            df.groupby("date")
            .agg(
                {
                    "total_hours_charged": "sum",
                    "direct_hours": "sum",
                    "overtime_hours": "sum",
                    "userid": "nunique",  # count unique employees
                    "fiscal_year": "first",
                    "fiscal_period": "first",
                }
            )
            .reset_index()
        )

        daily_metrics.rename(columns={"userid": "employee_count"}, inplace=True)

        # Calculate efficiency metrics
        daily_metrics["labor_efficiency"] = (
            daily_metrics["direct_hours"] / daily_metrics["total_hours_charged"]
        )

        return daily_metrics, dept_daily_metrics

    def create_manufacturing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create manufacturing-specific features"""
        df["is_month_end"] = df["date"].dt.is_month_end
        df["is_quarter_end"] = df["date"].dt.is_quarter_end

        # Calculate the last day of the current quarter for each date
        df["quarter_end_date"] = df["date"].apply(
            lambda x: pd.Timestamp(x.year, ((x.month - 1) // 3 + 1) * 3, 1)
            + pd.offsets.MonthEnd(0)
        )

        # Calculate days to quarter end
        df["days_to_quarter_end"] = (df["quarter_end_date"] - df["date"]).dt.days

        # Flag for typical maintenance periods
        # Assuming maintenance often happens at month-end
        df["is_maintenance_period"] = df["is_month_end"]

        # Clean up temporary columns
        df = df.drop("quarter_end_date", axis=1)

        return df
