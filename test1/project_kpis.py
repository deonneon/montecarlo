# project_kpis.py
import pandas as pd
import numpy as np


class ProjectKPIs:
    def __init__(self, df):
        self.df = df.copy()
        self._prepare_date_columns()

    def _prepare_date_columns(self):
        """Prepare date-related columns for aggregation"""
        self.df["week"] = self.df["date"].dt.isocalendar().week
        self.df["month"] = self.df["date"].dt.to_period("M")
        self.df["year"] = self.df["date"].dt.year

    def calculate_utilization_metrics(self, group):
        """Calculate basic utilization metrics for a group"""
        return pd.Series(
            {
                "direct_labor_ratio": (
                    group["direct_hours"].sum() / group["total_hours_charged"].sum()
                )
                * 100,
                "overtime_ratio": (
                    group["overtime_hours"].sum() / group["total_hours_charged"].sum()
                )
                * 100,
                "avg_daily_hours": group["total_hours_charged"].mean(),
                "employee_count": group["userid"].nunique(),
                "total_hours": group["total_hours_charged"].sum(),
            }
        )

    def get_monthly_kpis(self):
        """Calculate monthly KPIs by department"""
        return self.df.groupby(["year", "month", "dept"]).apply(
            self.calculate_utilization_metrics
        )

    def get_efficiency_kpis(self):
        """Calculate workforce efficiency KPIs"""
        efficiency = pd.DataFrame()

        efficiency["hours_per_employee"] = (
            self.df.groupby(["year", "month"])["total_hours_charged"].sum()
            / self.df.groupby(["year", "month"])["userid"].nunique()
        )

        overtime_metrics = (
            self.df[self.df["overtime_hours"] > 0]
            .groupby(["year", "month"])
            .agg({"userid": "nunique", "overtime_hours": ["sum", "mean"]})
        )

        return {"efficiency_metrics": efficiency, "overtime_metrics": overtime_metrics}

    def get_department_performance(self):
        """Calculate department performance metrics"""
        return self.df.groupby("dept").agg(
            {
                "total_hours_charged": ["mean", "std", "sum"],
                "direct_hours": "sum",
                "overtime_hours": "sum",
                "userid": "nunique",
            }
        )

    def get_weekly_performance(self, target_hours=40):
        """Calculate weekly performance vs target"""
        weekly = (
            self.df.groupby(["year", "week", "userid"])
            .agg({"total_hours_charged": "sum"})
            .reset_index()
        )

        weekly["hours_variance"] = weekly["total_hours_charged"] - target_hours
        return weekly

    def get_utilization_trends(self):
        """Calculate utilization trends"""
        trends = self.df.groupby(["year", "month"]).agg(
            {
                "direct_hours": "sum",
                "non_direct_hours": "sum",
                "total_hours_charged": "sum",
                "userid": "nunique",
            }
        )

        trends["utilization_rate"] = (
            trends["direct_hours"] / trends["total_hours_charged"] * 100
        )
        return trends

    def get_holiday_impact(self):
        """Analyze holiday impact on labor hours"""
        return self.df.groupby("is_holiday").agg(
            {
                "total_hours_charged": ["mean", "std"],
                "direct_hours": "mean",
                "overtime_hours": "mean",
            }
        )

    def generate_summary(self):
        """Generate a summary of all KPIs"""
        trends = self.get_utilization_trends()
        weekly_perf = self.get_weekly_performance()
        dept_perf = self.get_department_performance()

        return {
            "overall_metrics": {
                "avg_utilization_rate": trends["utilization_rate"].mean(),
                "overtime_percentage": (
                    weekly_perf["hours_variance"][
                        weekly_perf["hours_variance"] > 0
                    ].count()
                    / len(weekly_perf)
                    * 100
                ),
                "dept_efficiency": dept_perf["total_hours_charged"]["mean"].to_dict(),
            },
            "trending_metrics": {
                "utilization_trend": trends["utilization_rate"].tolist()[-6:],
                "recent_efficiency": self.get_efficiency_kpis()["efficiency_metrics"][
                    "hours_per_employee"
                ].tolist()[-6:],
            },
        }
