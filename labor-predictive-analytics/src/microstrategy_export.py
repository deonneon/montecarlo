import pandas as pd
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import os


class MicroStrategyExporter:
    def __init__(self):
        self.date_format = "%Y-%m-%d"

    def convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON serializable format"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {
                key: self.convert_to_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        return obj

    def export_to_json(self, data: Dict, filepath: str) -> None:
        """
        Export data to JSON file for MicroStrategy import
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Convert data to JSON serializable format
        serializable_data = self.convert_to_serializable(data)

        try:
            with open(filepath, "w") as f:
                json.dump(serializable_data, f, indent=2)
            print(f"Successfully exported data to {filepath}")
        except Exception as e:
            print(f"Error exporting data to {filepath}: {str(e)}")

    def prepare_labor_data(self, df: pd.DataFrame) -> Dict:
        """Prepare core labor data"""
        df["date"] = pd.to_datetime(df["date"]).dt.strftime(self.date_format)

        return {
            "metrics": {
                "total_hours_charged": df["total_hours_charged"].tolist(),
                "direct_hours": df["direct_hours"].tolist(),
                "overtime_hours": df["overtime_hours"].tolist(),
                "non_direct_hours": (
                    df["non_direct_hours"].tolist()
                    if "non_direct_hours" in df.columns
                    else []
                ),
                "employee_count": (
                    df["employee_count"].tolist()
                    if "employee_count" in df.columns
                    else []
                ),
                "labor_efficiency": (
                    df["labor_efficiency"].tolist()
                    if "labor_efficiency" in df.columns
                    else []
                ),
            },
            "attributes": {
                "date": df["date"].tolist(),
                "dept": df["dept"].unique().tolist() if "dept" in df.columns else [],
                "fiscal_year": df["fiscal_year"].unique().tolist(),
                "fiscal_period": list(range(1, 13)),
            },
        }

    def prepare_department_metrics(self, dept_data: pd.DataFrame) -> Dict:
        """Prepare department-level metrics"""
        return {
            "department_metrics": {
                dept: {
                    "total_hours": dept_data[dept_data["dept"] == dept][
                        "total_hours_charged"
                    ].tolist(),
                    "direct_hours": dept_data[dept_data["dept"] == dept][
                        "direct_hours"
                    ].tolist(),
                    "overtime_hours": dept_data[dept_data["dept"] == dept][
                        "overtime_hours"
                    ].tolist(),
                    "employee_count": (
                        dept_data[dept_data["dept"] == dept]["employee_count"].tolist()
                        if "employee_count" in dept_data.columns
                        else []
                    ),
                }
                for dept in dept_data["dept"].unique()
            }
        }

    def prepare_worker_metrics(self, worker_data: pd.DataFrame) -> Dict:
        """Prepare worker-level metrics"""
        return {
            "worker_metrics": {
                worker: {
                    "total_hours": worker_data[worker_data["userid"] == worker][
                        "total_hours_charged"
                    ].tolist(),
                    "direct_hours": worker_data[worker_data["userid"] == worker][
                        "direct_hours"
                    ].tolist(),
                    "overtime_hours": worker_data[worker_data["userid"] == worker][
                        "overtime_hours"
                    ].tolist(),
                }
                for worker in worker_data["userid"].unique()
            }
        }

    def prepare_forecast_data(self, forecast_results: Dict) -> Dict:
        """Prepare all forecast-related data"""
        return {
            "aggregate_forecast": {
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
            },
            "worker_forecasts": forecast_results.get("worker_forecasts", {}),
            "hybrid_forecast": forecast_results.get("hybrid_forecast", {}),
        }

    def prepare_kpi_data(self, kpi_results: Dict) -> Dict:
        """Prepare KPI metrics"""
        return {
            "kpi_metrics": {
                "labor_efficiency": kpi_results.get("labor_efficiency", []),
                "overtime_ratio": kpi_results.get("overtime_ratio", []),
                "productivity_index": kpi_results.get("productivity_index", []),
            }
        }

    def prepare_fiscal_analysis(self, fiscal_data: Dict) -> Dict:
        """Prepare fiscal year analysis data"""
        return {
            "fiscal_analysis": {
                "yearly_patterns": fiscal_data.get("yearly_patterns", {}),
                "period_trends": fiscal_data.get("period_trends", {}),
                "year_over_year": fiscal_data.get("year_over_year", {}),
            }
        }

    def export_all_data(
        self,
        daily_data: pd.DataFrame,
        dept_data: pd.DataFrame,
        worker_data: pd.DataFrame,
        forecast_results: Dict,
        kpi_results: Dict,
        fiscal_data: Dict,
    ) -> None:
        """Export all dashboard components"""

        complete_export = {
            "labor_data": self.prepare_labor_data(daily_data),
            "department_metrics": self.prepare_department_metrics(dept_data),
            "worker_metrics": self.prepare_worker_metrics(worker_data),
            "forecasts": self.prepare_forecast_data(forecast_results),
            "kpis": self.prepare_kpi_data(kpi_results),
            "fiscal_analysis": self.prepare_fiscal_analysis(fiscal_data),
        }

        # Export to single JSON file
        self.export_to_json(
            complete_export, "data/microstrategy/complete_dashboard_data.json"
        )

        # Generate SQL queries
        self.export_sql_queries()

    def export_sql_queries(self) -> None:
        """Export all necessary SQL queries"""
        queries = {
            "labor_data": """
                SELECT 
                    date, userid, dept,
                    total_hours_charged, direct_hours,
                    non_direct_hours, overtime_hours,
                    fiscal_year, fiscal_period
                FROM labor_data
                WHERE date BETWEEN @start_date AND @end_date
            """,
            "department_metrics": """
                SELECT 
                    dept, date,
                    SUM(total_hours_charged) as total_hours,
                    SUM(direct_hours) as direct_hours,
                    SUM(overtime_hours) as overtime_hours,
                    COUNT(DISTINCT userid) as employee_count
                FROM labor_data
                GROUP BY dept, date
            """,
            "worker_metrics": """
                SELECT 
                    userid, date,
                    total_hours_charged,
                    direct_hours,
                    overtime_hours
                FROM labor_data
                WHERE userid = @worker_id
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
            """,
        }

        with open("data/microstrategy/queries.sql", "w") as f:
            for name, query in queries.items():
                f.write(f"-- {name}\n{query}\n\n")
