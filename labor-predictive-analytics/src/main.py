from data_processing import DataPreprocessor
from time_series_model import TimeSeriesPredictor
from monte_carlo import MonteCarloSimulator
from kpi_calculation import KPICalculator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from microstrategy_export import MicroStrategyExporter
from worker_monte_carlo import WorkerMonteCarloPredictor
from hybrid_predictor import HybridLaborPredictor


def export_for_microstrategy(daily_data, forecast_results):
    """Export data in MicroStrategy-compatible format"""
    try:
        exporter = MicroStrategyExporter()

        # Create directory if it doesn't exist
        os.makedirs("data/microstrategy", exist_ok=True)

        # Prepare and export labor data
        labor_data = exporter.prepare_labor_data(daily_data)
        exporter.export_to_json(labor_data, "data/microstrategy/labor_data.json")

        # Prepare and export forecast data
        forecast_data = exporter.prepare_forecast_data(forecast_results)
        exporter.export_to_json(forecast_data, "data/microstrategy/forecast_data.json")

        # Generate SQL queries
        sql_queries = exporter.generate_sql_queries()
        with open("data/microstrategy/queries.sql", "w") as f:
            for name, query in sql_queries.items():
                f.write(f"-- {name}\n{query}\n\n")

        print("Successfully exported data for MicroStrategy")

    except Exception as e:
        print(f"Error exporting data for MicroStrategy: {str(e)}")


def main():
    # Initialize components with fiscal year settings
    preprocessor = DataPreprocessor(fiscal_start_month=10)  # October start
    predictor = TimeSeriesPredictor(
        yearly_seasonal_periods=252,  # Working days in a year
        weekly_seasonal_periods=5,  # 5-day work week
    )
    simulator = MonteCarloSimulator(n_simulations=1000)
    kpi_calc = KPICalculator()
    worker_predictor = WorkerMonteCarloPredictor()
    hybrid_predictor = HybridLaborPredictor()  # Add this line

    # Load and prepare data
    raw_data = preprocessor.load_and_prepare_data("data/generated/labor_data.csv")

    # Unpack the two DataFrames returned by aggregate_daily_metrics
    daily_data, dept_daily_data = preprocessor.aggregate_daily_metrics(raw_data)
    daily_data = preprocessor.create_manufacturing_features(daily_data)

    # Get fiscal year pattern
    fiscal_pattern = predictor.get_fiscal_year_pattern(daily_data)

    # Fit time series model
    ts_data = daily_data["total_hours_charged"].values
    predictor.fit_holtwinters(ts_data)

    # Generate Monte Carlo simulations
    forecast_horizon = 252  # One fiscal year ahead
    mean_forecast, lower_bound, upper_bound = simulator.generate_scenarios(
        predictor.model, daily_data, forecast_horizon, fiscal_pattern
    )

    # Generate worker predictions
    worker_predictions = worker_predictor.predict_all_workers(
        raw_data, forecast_horizon
    )

    # Generate hybrid forecast
    hybrid_mean, hybrid_lower, hybrid_upper = hybrid_predictor.generate_hybrid_forecast(
        raw_data, forecast_horizon, include_seasonality=True, include_growth=True
    )

    # Calculate KPIs
    labor_kpis = kpi_calc.calculate_labor_efficiency(daily_data)

    # Visualize results
    plot_results(daily_data, mean_forecast, lower_bound, upper_bound)
    plot_fiscal_pattern(fiscal_pattern)

    # Export for MicroStrategy
    forecast_results = {
        "mean_forecast": mean_forecast,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "start_date": daily_data["date"].max().strftime("%Y-%m-%d"),
        "worker_forecasts": worker_predictions,
        "hybrid_forecast": {
            "mean": hybrid_mean,
            "lower_bound": hybrid_lower,
            "upper_bound": hybrid_upper,
        },
    }

    fiscal_data = {
        "yearly_patterns": (
            fiscal_pattern.to_dict() if isinstance(fiscal_pattern, pd.DataFrame) else {}
        ),
        "period_trends": daily_data.groupby("fiscal_period")["total_hours_charged"]
        .mean()
        .to_dict(),
        "year_over_year": daily_data.groupby("fiscal_year")["total_hours_charged"]
        .sum()
        .to_dict(),
    }

    # Export all data to MicroStrategy
    exporter = MicroStrategyExporter()
    exporter.export_all_data(
        daily_data=daily_data,
        dept_data=dept_daily_data,
        worker_data=raw_data,
        forecast_results=forecast_results,
        kpi_results=labor_kpis,
        fiscal_data=fiscal_data,
    )

    return labor_kpis


def plot_results(daily_data, mean_forecast, lower_bound, upper_bound):
    plt.figure(figsize=(15, 7))

    # Plot historical data
    plt.plot(
        daily_data["date"],
        daily_data["total_hours_charged"],
        label="Historical Data",
        color="blue",
    )

    # Plot forecast
    forecast_dates = pd.date_range(
        start=daily_data["date"].iloc[-1], periods=len(mean_forecast) + 1
    )[1:]

    plt.plot(
        forecast_dates, mean_forecast, label="Forecast", color="red", linestyle="--"
    )
    plt.fill_between(
        forecast_dates,
        lower_bound,
        upper_bound,
        color="red",
        alpha=0.2,
        label="95% Confidence Interval",
    )

    # Add fiscal year boundaries
    fiscal_years = daily_data["fiscal_year"].unique()
    for fy in fiscal_years:
        fy_start = pd.Timestamp(f"{fy}-10-01")
        if fy_start >= daily_data["date"].min() and fy_start <= forecast_dates[-1]:
            plt.axvline(x=fy_start, color="gray", linestyle=":", alpha=0.5)
            plt.text(fy_start, plt.ylim()[1], f"FY{fy}", rotation=90)

    plt.title("Labor Hours Forecast with Monte Carlo Simulation (By Fiscal Year)")
    plt.xlabel("Date")
    plt.ylabel("Total Hours Charged")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_fiscal_pattern(fiscal_pattern):
    plt.figure(figsize=(12, 6))

    # Plot mean hours by fiscal period
    plt.bar(
        fiscal_pattern["fiscal_period"],
        fiscal_pattern["mean"],
        yerr=fiscal_pattern["std"],
        capsize=5,
    )

    plt.title("Average Labor Hours by Fiscal Period")
    plt.xlabel("Fiscal Period (1 = October)")
    plt.ylabel("Average Hours")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
