import pandas as pd
import json
from datetime import datetime
from typing import Dict, List


class MicroStrategyExporter:
    def __init__(self):
        self.date_format = "%Y-%m-%d"

    def prepare_labor_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare labor data in MicroStrategy-compatible format
        """
        # Convert date to MicroStrategy format
        df["date"] = pd.to_datetime(df["date"]).dt.strftime(self.date_format)

        # Create metrics structure
        metrics = {
            "total_hours_charged": df["total_hours_charged"].tolist(),
            "direct_hours": df["direct_hours"].tolist(),
            "overtime_hours": df["overtime_hours"].tolist(),
            "employee_count": df["userid"].nunique(),
        }

        # Create attributes structure
        attributes = {
            "date": df["date"].tolist(),
            "dept": df["dept"].unique().tolist(),
            "fiscal_year": df["fiscal_year"].unique().tolist(),
            "fiscal_period": list(range(1, 13)),
        }

        return {"metrics": metrics, "attributes": attributes}

    def prepare_forecast_data(self, forecast_results: Dict) -> Dict:
        """
        Prepare forecast data for MicroStrategy
        """
        return {
            "forecast": {
                "mean": forecast_results["mean_forecast"].tolist(),
                "lower_bound": forecast_results["lower_bound"].tolist(),
                "upper_bound": forecast_results["upper_bound"].tolist(),
                "dates": [
                    (
                        datetime.strptime(
                            forecast_results["start_date"], self.date_format
                        )
                        + pd.Timedelta(days=x)
                    ).strftime(self.date_format)
                    for x in range(len(forecast_results["mean_forecast"]))
                ],
            }
        }

    def export_to_json(self, data: Dict, filepath: str) -> None:
        """
        Export data to JSON file for MicroStrategy import
        """
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def generate_sql_queries(self) -> Dict[str, str]:
        """
        Generate SQL queries for MicroStrategy
        """
        queries = {
            "labor_data": """
                SELECT 
                    date,
                    dept,
                    userid,
                    total_hours_charged,
                    direct_hours,
                    overtime_hours,
                    fiscal_year,
                    fiscal_period
                FROM labor_data
                WHERE date BETWEEN @start_date AND @end_date
            """,
            "department_summary": """
                SELECT 
                    dept,
                    date,
                    SUM(total_hours_charged) as total_hours,
                    SUM(direct_hours) as direct_hours,
                    SUM(overtime_hours) as overtime_hours,
                    COUNT(DISTINCT userid) as employee_count
                FROM labor_data
                WHERE date BETWEEN @start_date AND @end_date
                GROUP BY dept, date
            """,
            "fiscal_metrics": """
                SELECT 
                    fiscal_year,
                    fiscal_period,
                    SUM(total_hours_charged) as total_hours,
                    AVG(total_hours_charged) as avg_hours,
                    COUNT(DISTINCT userid) as employee_count
                FROM labor_data
                GROUP BY fiscal_year, fiscal_period
                ORDER BY fiscal_year, fiscal_period
            """,
        }
        return queries
