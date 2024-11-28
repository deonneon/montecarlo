import pandas as pd
import numpy as np
from typing import Tuple, Optional


class DataPreprocessor:
    def __init__(self, fiscal_start_month: int = 10):
        self.fiscal_start_month = fiscal_start_month

    def load_and_prepare_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare data for time series analysis"""
        df = pd.read_csv(filepath)
        df["date"] = pd.to_datetime(df["date"])

        # Ensure is_holiday is properly loaded as boolean
        if "is_holiday" not in df.columns:
            # If holiday data isn't in the CSV, we'll add it based on the dates
            holidays = [
                "2023-01-01",  # New Year's Day
                "2023-01-16",  # Martin Luther King Jr. Day
                "2023-02-20",  # Presidents' Day
                "2023-05-29",  # Memorial Day
                "2023-06-19",  # Juneteenth
                "2023-07-04",  # Independence Day
                "2023-09-04",  # Labor Day
                "2023-10-09",  # Columbus Day
                "2023-11-10",  # Veterans Day (observed)
                "2023-11-23",  # Thanksgiving Day
                "2023-12-25",  # Christmas Day
            ]
            df["is_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(holidays)

        # Add fiscal year and period columns
        df["fiscal_year"] = df["date"].apply(
            lambda x: x.year if x.month >= self.fiscal_start_month else x.year - 1
        )
        df["fiscal_period"] = df["date"].apply(
            lambda x: (x.month - self.fiscal_start_month) % 12 + 1
        )

        return df

    def aggregate_daily_metrics(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate data to daily level with manufacturing-specific metrics"""
        # Department-level aggregation
        dept_daily_metrics = (
            df.groupby(["date", "dept", "is_holiday"])
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

        # Overall daily metrics
        daily_metrics = (
            df.groupby(["date", "is_holiday"])
            .agg(
                {
                    "total_hours_charged": "sum",
                    "direct_hours": "sum",
                    "overtime_hours": "sum",
                    "userid": "nunique",
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
