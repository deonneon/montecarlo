# -*- coding: utf-8 -*-
import dataiku
import pandas as pd
from datetime import datetime

# Get handles on input/output datasets
labor_data = dataiku.Dataset("labor_data")
daily_metrics_output = dataiku.Dataset("daily_metrics")
monthly_dept_metrics_output = dataiku.Dataset("monthly_dept_metrics")
employee_metrics_output = dataiku.Dataset("employee_metrics")
kpi_metrics_output = dataiku.Dataset("kpi_metrics")

# Read input data
df = labor_data.get_dataframe(parse_dates=["date"])

# 1. Daily Employee Metrics
daily_metrics = pd.DataFrame(
    {
        "date": df["date"],
        "employee_count": df.groupby("date")["userid"].transform("nunique"),
        "total_hours": df.groupby("date")["total_hours_charged"].transform("sum"),
        "total_overtime": df.groupby("date")["overtime_hours"].transform("sum"),
        "total_direct_hours": df.groupby("date")["direct_hours"].transform("sum"),
        "total_non_direct_hours": df.groupby("date")["non_direct_hours"].transform(
            "sum"
        ),
    }
).drop_duplicates()

# 2. Monthly Department Metrics
monthly_dept_metrics = (
    df.groupby([pd.Grouper(key="date", freq="M"), "dept"])
    .agg(
        {
            "userid": "nunique",
            "total_hours_charged": "sum",
            "overtime_hours": "sum",
            "direct_hours": "sum",
            "non_direct_hours": "sum",
        }
    )
    .reset_index()
)
monthly_dept_metrics.columns = [
    "month",
    "department",
    "employee_count",
    "total_hours",
    "overtime_hours",
    "direct_hours",
    "non_direct_hours",
]

# 3. Employee Performance Metrics
employee_metrics = (
    df.groupby(["userid", "dept"])
    .agg(
        {
            "total_hours_charged": "mean",
            "overtime_hours": "mean",
            "direct_hours": "mean",
            "non_direct_hours": "mean",
        }
    )
    .reset_index()
)
employee_metrics.columns = [
    "userid",
    "department",
    "avg_hours",
    "avg_overtime",
    "avg_direct_hours",
    "avg_non_direct_hours",
]

# Time period calculations for KPIs
today = df["date"].max()
current_year = today.year
previous_year = current_year - 1
current_day_of_month = today.day

# MTD Calculations
df_mtd_current = df[
    (df["date"].dt.year == current_year) & (df["date"].dt.day <= current_day_of_month)
]
df_mtd_previous = df[
    (df["date"].dt.year == previous_year) & (df["date"].dt.day <= current_day_of_month)
]

# QTD Calculations
current_quarter = today.to_period("Q")
df_qtd_current = df[df["date"].dt.to_period("Q") == current_quarter]
previous_year_quarter = (today - pd.DateOffset(years=1)).to_period("Q")
df_qtd_previous = df[df["date"].dt.to_period("Q") == previous_year_quarter]

# YTD Calculations
current_day_of_year = today.dayofyear
df_ytd_current = df[
    (df["date"].dt.year == current_year)
    & (df["date"].dt.dayofyear <= current_day_of_year)
]
df_ytd_previous = df[
    (df["date"].dt.year == previous_year)
    & (df["date"].dt.dayofyear <= current_day_of_year)
]

# 4. Consolidated KPI DataFrame
kpi_metrics = pd.DataFrame(
    {
        "date": [today],
        "mtd_employees": [df_mtd_current["userid"].nunique()],
        "mtd_employees_prev_year": [df_mtd_previous["userid"].nunique()],
        "qtd_employees": [df_qtd_current["userid"].nunique()],
        "qtd_employees_prev_year": [df_qtd_previous["userid"].nunique()],
        "ytd_employees": [df_ytd_current["userid"].nunique()],
        "ytd_employees_prev_year": [df_ytd_previous["userid"].nunique()],
        "last_updated": [datetime.now()],
    }
)

# Write output datasets
daily_metrics_output.write_with_schema(daily_metrics)
monthly_dept_metrics_output.write_with_schema(monthly_dept_metrics)
employee_metrics_output.write_with_schema(employee_metrics)
kpi_metrics_output.write_with_schema(kpi_metrics)
