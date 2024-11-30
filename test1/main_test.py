import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os

warnings.filterwarnings("ignore")

from project_kpis import ProjectKPIs


# 1. Data Ingestion and Initial Processing
def load_and_prepare_data():
    df = pd.read_csv("labor_data.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


def create_train_test_split(df):
    # Sort data by date and get the last 30 days as test set
    df_sorted = df.sort_values("date")
    test_start_date = df_sorted["date"].max() - pd.Timedelta(days=30)

    train_df = df[df["date"] < test_start_date]
    test_df = df[df["date"] >= test_start_date]

    return train_df, test_df


def evaluate_forecasts(actual, predicted, method_name):
    """Calculate error metrics for the forecasts"""
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae = np.mean(np.abs(actual - predicted))

    return {"method": method_name, "MAPE": mape, "RMSE": rmse, "MAE": mae}


def create_comparison_plots(
    test_df, prophet_overall_pred, prophet_employee_pred, monte_carlo_pred
):
    # Prepare actual values
    actual = test_df.groupby("date")["total_hours_charged"].sum()
    dates = actual.index

    # Create the main figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: All predictions vs actual
    ax1.plot(dates, actual.values, "k-", label="Actual", linewidth=2)
    ax1.plot(dates, prophet_overall_pred, "b--", label="Prophet Overall", alpha=0.7)
    ax1.plot(
        dates, prophet_employee_pred, "r--", label="Prophet by Employee", alpha=0.7
    )
    ax1.plot(dates, monte_carlo_pred, "g--", label="Monte Carlo", alpha=0.7)

    ax1.set_title("Comparison of All Methods", fontsize=14)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Total Hours", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Error comparison
    error_prophet_overall = prophet_overall_pred - actual.values
    error_prophet_employee = prophet_employee_pred - actual.values
    error_monte_carlo = monte_carlo_pred - actual.values

    ax2.plot(
        dates, error_prophet_overall, "b-", label="Prophet Overall Error", alpha=0.7
    )
    ax2.plot(
        dates,
        error_prophet_employee,
        "r-",
        label="Prophet by Employee Error",
        alpha=0.7,
    )
    ax2.plot(dates, error_monte_carlo, "g-", label="Monte Carlo Error", alpha=0.7)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.3)

    ax2.set_title("Prediction Errors by Method", fontsize=14)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.set_ylabel("Error (Predicted - Actual)", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("forecast_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create error distribution plot
    plt.figure(figsize=(12, 6))
    plt.hist(
        error_prophet_overall, bins=20, alpha=0.3, label="Prophet Overall", color="blue"
    )
    plt.hist(
        error_prophet_employee,
        bins=20,
        alpha=0.3,
        label="Prophet by Employee",
        color="red",
    )
    plt.hist(error_monte_carlo, bins=20, alpha=0.3, label="Monte Carlo", color="green")

    plt.title("Distribution of Prediction Errors", fontsize=14)
    plt.xlabel("Error (Predicted - Actual)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig("error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


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


# 3. Time Series Analysis using Prophet
def perform_prophet_analysis(train_df, test_df):
    # Prepare data for Prophet
    prophet_data = train_df.groupby("date")["total_hours_charged"].sum().reset_index()
    prophet_data.columns = ["ds", "y"]

    # Train Prophet model
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
    )
    model.add_country_holidays(country_name="US")
    model.fit(prophet_data)

    # Predict for test period
    future_dates = pd.DataFrame({"ds": test_df["date"].unique()})
    forecast = model.predict(future_dates)

    # Get actual values for test period
    actual = test_df.groupby("date")["total_hours_charged"].sum()

    return {
        "metrics": evaluate_forecasts(
            actual.values, forecast["yhat"].values, "Prophet Overall"
        ),
        "predictions": forecast["yhat"].values,
    }


def perform_employee_prophet_analysis(train_df, test_df):
    employee_forecasts = []

    for employee in train_df["userid"].unique():
        # Train data for this employee
        emp_train = train_df[train_df["userid"] == employee]
        emp_data = emp_train.groupby("date")["total_hours_charged"].sum().reset_index()
        emp_data.columns = ["ds", "y"]

        # Train model
        model = Prophet(
            yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=True
        )
        model.add_country_holidays(country_name="US")
        model.fit(emp_data)

        # Predict
        future_dates = pd.DataFrame({"ds": test_df["date"].unique()})
        forecast = model.predict(future_dates)
        forecast["employee"] = employee
        employee_forecasts.append(forecast)

    # Combine all forecasts
    all_forecasts = pd.concat(employee_forecasts)

    # Aggregate predictions
    aggregated_forecast = all_forecasts.groupby("ds")["yhat"].sum()

    # Get actual values
    actual = test_df.groupby("date")["total_hours_charged"].sum()

    return {
        "metrics": evaluate_forecasts(
            actual.values, aggregated_forecast.values, "Prophet by Employee"
        ),
        "predictions": aggregated_forecast.values,
    }


def perform_monte_carlo(train_df, test_df, n_simulations=1000):
    # Create a holiday flag in both training and test data
    from pandas.tseries.holiday import USFederalHolidayCalendar

    cal = USFederalHolidayCalendar()

    # Get all holidays for the date range
    holidays = cal.holidays(start=train_df["date"].min(), end=test_df["date"].max())

    # Add holiday flag to training data
    train_df = train_df.copy()
    train_df["is_holiday"] = train_df["date"].isin(holidays)

    # Add holiday flag to test data
    test_df = test_df.copy()
    test_df["is_holiday"] = test_df["date"].isin(holidays)

    # Calculate employee statistics separately for holidays and non-holidays
    employee_stats = (
        train_df.groupby(["userid", "dept", "is_holiday"])
        .agg({"total_hours_charged": ["mean", "std"]})
        .reset_index()
    )

    # Initialize array for simulations
    test_dates = test_df["date"].unique()
    simulated_totals = np.zeros((n_simulations, len(test_dates)))

    # Perform simulations
    for sim in range(n_simulations):
        daily_totals = np.zeros(len(test_dates))

        for date_idx, date in enumerate(test_dates):
            is_holiday = test_df[test_df["date"] == date]["is_holiday"].iloc[0]

            # Filter employee stats for the current holiday status
            current_stats = employee_stats[employee_stats["is_holiday"] == is_holiday]

            # Generate hours for each employee
            for _, emp_stats in current_stats.iterrows():
                emp_hours = np.random.normal(
                    emp_stats[("total_hours_charged", "mean")],
                    emp_stats[("total_hours_charged", "std")],
                )
                emp_hours = max(emp_hours, 0)  # Ensure non-negative hours
                daily_totals[date_idx] += emp_hours

        simulated_totals[sim] = daily_totals

    # Use mean of simulations as prediction
    predictions = np.mean(simulated_totals, axis=0)

    # Get actual values
    actual = test_df.groupby("date")["total_hours_charged"].sum()

    return {
        "metrics": evaluate_forecasts(actual.values, predictions, "Monte Carlo"),
        "predictions": predictions,
    }


def main():
    # Load data
    print("Loading data...")
    df = load_and_prepare_data()

    # Create train-test split
    train_df, test_df = create_train_test_split(df)
    print(f"Training data up to: {train_df['date'].max()}")
    print(f"Test period: {test_df['date'].min()} to {test_df['date'].max()}")

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

    # Evaluate all methods
    print("\nEvaluating Prophet overall...")
    prophet_overall_results = perform_prophet_analysis(train_df, test_df)

    print("Evaluating Prophet by employee...")
    prophet_employee_results = perform_employee_prophet_analysis(train_df, test_df)

    print("Evaluating Monte Carlo simulation...")
    monte_carlo_results = perform_monte_carlo(train_df, test_df)

    # Create results DataFrame
    results = [
        prophet_overall_results["metrics"],
        prophet_employee_results["metrics"],
        monte_carlo_results["metrics"],
    ]
    results_df = pd.DataFrame(results)
    print("\nBenchmarking Results:")
    print(results_df)

    # Create visualizations
    create_comparison_plots(
        test_df,
        prophet_overall_results["predictions"],
        prophet_employee_results["predictions"],
        monte_carlo_results["predictions"],
    )

    # Save results
    results_df.to_csv("forecasting_benchmark_results.csv", index=False)

    print("\nBenchmarking complete. Results saved to:")
    print("- 'forecasting_benchmark_results.csv'")
    print("- 'forecast_comparison.png'")
    print("- 'error_distribution.png'")


if __name__ == "__main__":
    main()
