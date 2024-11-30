import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

from project_kpis import ProjectKPIs


# 1. Data Ingestion and Initial Processing
def load_and_prepare_data():
    df = pd.read_csv("labor_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


# 2. Exploratory Data Analysis
def perform_eda(df):
    # Weekly aggregation
    weekly_data = (
        df.groupby([pd.Grouper(key="date", freq="W"), "dept"])
        .agg(
            {
                "total_hours_charged": "sum",
                "direct_hours": "sum",
                "non_direct_hours": "sum",
                "overtime_hours": "sum",
                "userid": "nunique",
            }
        )
        .reset_index()
    )

    weekly_data.rename(columns={"userid": "employee_count"}, inplace=True)

    # Calculate department-wise metrics
    dept_metrics = df.groupby("dept").agg(
        {
            "total_hours_charged": ["mean", "std"],
            "direct_hours": "mean",
            "overtime_hours": "mean",
            "userid": "nunique",
        }
    )

    return weekly_data, dept_metrics


# Prophet (both versions):

# Generally reliable for short to medium-term forecasts (1-3 months)
# Can forecast further but accuracy deteriorates significantly beyond 3-6 months
# Best used when you have strong seasonal patterns and at least 2 years of historical data


# 3. Time Series Analysis using Prophet
def perform_prophet_analysis(df):
    # Prepare data for Prophet (total hours by day)
    prophet_data = df.groupby("date")["total_hours_charged"].sum().reset_index()
    prophet_data.columns = ["ds", "y"]

    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
    )
    model.add_country_holidays(country_name="US")
    model.fit(prophet_data)

    # Make future predictions (30 days ahead)
    future_dates = model.make_future_dataframe(periods=30, freq="D")
    forecast = model.predict(future_dates)

    return forecast


def perform_employee_prophet_analysis(df):
    employee_forecasts = []

    for employee in df["userid"].unique():
        # Filter data for this employee (keeping daily data)
        emp_data = (
            df[df["userid"] == employee]
            .groupby("date")["total_hours_charged"]
            .sum()
            .reset_index()
        )

        emp_data.columns = ["ds", "y"]

        # Train Prophet model for this employee
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
        )
        model.add_country_holidays(country_name="US")
        model.fit(emp_data)

        # Make future predictions (30 days ahead)
        future_dates = model.make_future_dataframe(periods=30, freq="D")
        emp_forecast = model.predict(future_dates)

        # Add employee identifier
        emp_forecast["employee"] = employee

        # Append to list
        employee_forecasts.append(emp_forecast)

    # Combine all employee forecasts
    all_employee_forecasts = pd.concat(employee_forecasts, ignore_index=True)

    # Create aggregated forecast
    aggregated_forecast = (
        all_employee_forecasts.groupby("ds")
        .agg({"yhat": "sum", "yhat_lower": "sum", "yhat_upper": "sum"})
        .reset_index()
    )

    return {
        "employee_forecasts": all_employee_forecasts,
        "aggregated_forecast": aggregated_forecast,
    }


# Monte Carlo Simulation:

# More focused on understanding uncertainty ranges than point predictions
# Best for shorter horizons (1-2 months) due to compounding uncertainty
# Becomes increasingly wide and less informative beyond that

# How to improve performance: Vectorize the calculations:
# Instead of looping through employees, you can vectorize the operations using NumPy's broadcasting capabilities.
# Or use parallel processing.


# 4. Monte Carlo Simulation
def perform_monte_carlo(df, n_simulations=1000):
    # First, let's create a holiday calendar for the simulation period
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from datetime import datetime, timedelta

    cal = USFederalHolidayCalendar()

    # Get the last date from df and create future dates for simulation
    last_date = df["date"].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=30, freq="D"
    )

    # Get holidays within the simulation period
    holidays = cal.holidays(start=future_dates[0], end=future_dates[-1])
    holiday_dates = set(holidays.date)

    # Get employee-level daily statistics, separate for holidays and regular days
    employee_stats = df.copy()
    employee_stats["is_holiday"] = employee_stats["date"].dt.date.isin(
        cal.holidays(start=df["date"].min(), end=df["date"].max()).date
    )

    # Calculate statistics for regular days and holidays
    employee_stats = (
        employee_stats.groupby(["userid", "dept", "is_holiday"])
        .agg(
            {
                "total_hours_charged": ["mean", "std"],
                "direct_hours": ["mean", "std"],
                "non_direct_hours": ["mean", "std"],
                "overtime_hours": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Initialize array to store simulation results
    simulated_totals = np.zeros((n_simulations, 30))

    # For each simulation
    for sim in range(n_simulations):
        daily_totals = np.zeros(30)

        # Simulate each employee's hours
        for userid in employee_stats["userid"].unique():
            emp_stats_regular = employee_stats[
                (employee_stats["userid"] == userid) & (~employee_stats["is_holiday"])
            ].iloc[0]

            emp_stats_holiday = employee_stats[
                (employee_stats["userid"] == userid) & (employee_stats["is_holiday"])
            ]

            # If no holiday data exists for this employee, use regular day stats with reduction
            if emp_stats_holiday.empty:
                emp_stats_holiday = emp_stats_regular.copy()
                # Assume reduced hours on holidays (you can adjust this factor)
                holiday_reduction_factor = 0.5
                emp_stats_holiday[
                    ("total_hours_charged", "mean")
                ] *= holiday_reduction_factor
                emp_stats_holiday[
                    ("total_hours_charged", "std")
                ] *= holiday_reduction_factor

            else:
                emp_stats_holiday = emp_stats_holiday.iloc[0]

            # Generate hours for each day in the simulation period
            for day_idx, date in enumerate(future_dates):
                is_holiday = date.date() in holiday_dates

                if is_holiday:
                    stats = emp_stats_holiday
                else:
                    stats = emp_stats_regular

                # Generate hours with appropriate distribution
                hours = np.random.normal(
                    stats[("total_hours_charged", "mean")],
                    stats[("total_hours_charged", "std")],
                )

                # Ensure no negative hours
                hours = max(hours, 0)

                # Add to daily totals
                daily_totals[day_idx] += hours

        simulated_totals[sim] = daily_totals

    # Calculate confidence intervals
    confidence_intervals = {
        "lower_95": np.percentile(simulated_totals, 2.5, axis=0),
        "upper_95": np.percentile(simulated_totals, 97.5, axis=0),
        "mean": np.mean(simulated_totals, axis=0),
        "dates": future_dates,
    }

    return confidence_intervals


# 5. Prepare data for visualization
def prepare_for_visualization(weekly_data, forecast, monte_carlo_results):
    # Combine actual and forecasted data
    viz_data = pd.DataFrame(
        {
            "date": forecast["ds"],
            "actual_hours": weekly_data.groupby("date")["total_hours_charged"].sum(),
            "forecast": forecast["yhat"],
            "forecast_lower": forecast["yhat_lower"],
            "forecast_upper": forecast["yhat_upper"],
        }
    )

    # Add Monte Carlo results for the future dates
    future_dates = pd.date_range(
        start=weekly_data["date"].max() + pd.Timedelta(days=1), periods=30, freq="D"
    )

    monte_carlo_df = pd.DataFrame(
        {
            "date": future_dates,
            "mc_mean": monte_carlo_results["mean"],
            "mc_lower": monte_carlo_results["lower_95"],
            "mc_upper": monte_carlo_results["upper_95"],
        }
    )

    return viz_data, monte_carlo_df


def main():
    # Load data
    print("Loading data...")
    df = load_and_prepare_data()
    df["date"] = pd.to_datetime(df["date"])

    # Initialize KPI calculator
    print("Calculating Project Management KPIs...")
    kpi_calculator = ProjectKPIs(df)

    # Generate KPI reports
    kpi_data = {
        "monthly_kpis": kpi_calculator.get_monthly_kpis(),
        "efficiency_kpis": kpi_calculator.get_efficiency_kpis()["efficiency_metrics"],
        "dept_performance": kpi_calculator.get_department_performance(),
        "weekly_performance": kpi_calculator.get_weekly_performance(),
        "utilization_trends": kpi_calculator.get_utilization_trends(),
        "holiday_impact": kpi_calculator.get_holiday_impact(),
    }

    # Get summary
    summary = kpi_calculator.generate_summary()

    # Save KPI data
    for name, data in kpi_data.items():
        data.to_csv(f"{name}.csv")

    # Print key findings
    print("\nKey Project Management Metrics:")
    print(
        f"Average Utilization Rate: {summary['overall_metrics']['avg_utilization_rate']:.2f}%"
    )
    print(
        f"Overtime Occurrence: {summary['overall_metrics']['overtime_percentage']:.2f}%"
    )
    print("\nDepartment Efficiency (Average Hours):")
    for dept, hours in summary["overall_metrics"]["dept_efficiency"].items():
        print(f"{dept}: {hours:.2f} hours")

    # Perform EDA
    print("Performing exploratory data analysis...")
    weekly_data, dept_metrics = perform_eda(df)
    print("\nDepartment Metrics:")
    print(dept_metrics)

    # Perform Prophet analysis
    print("\nPerforming Prophet analysis...")
    forecast = perform_prophet_analysis(df)

    # Create directory for employee forecasts if it doesn't exist
    import os

    os.makedirs("employee_forecasts", exist_ok=True)

    print("\nPerforming employee-level Prophet analysis...")
    employee_prophet_results = perform_employee_prophet_analysis(df)

    # Save as single CSV with all employee forecasts
    employee_prophet_results["employee_forecasts"].to_csv(
        "employee_prophet_forecasts.csv", index=False
    )
    employee_prophet_results["aggregated_forecast"].to_csv(
        "employee_prophet_aggregated.csv", index=False
    )

    # Perform Monte Carlo simulation
    print("Performing Monte Carlo simulation...")
    monte_carlo_results = perform_monte_carlo(df)

    # Prepare visualization data
    print("Preparing data for visualization...")
    viz_data, monte_carlo_df = prepare_for_visualization(
        weekly_data, forecast, monte_carlo_results
    )

    # Save processed data for visualization
    viz_data.to_csv("prophet_forecast_data.csv", index=False)
    monte_carlo_df.to_csv("monte_carlo_results.csv", index=False)
    weekly_data.to_csv("weekly_aggregated_data.csv", index=False)

    print("\nData processing complete. Files saved for visualization:")
    print("- prophet_forecast_data.csv")
    print("- monte_carlo_results.csv")
    print("- weekly_aggregated_data.csv")


if __name__ == "__main__":
    main()
