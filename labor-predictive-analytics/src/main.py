from data_processing import DataPreprocessor
from time_series_model import TimeSeriesPredictor
from monte_carlo import MonteCarloSimulator
from kpi_calculation import KPICalculator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    # Initialize components with fiscal year settings
    preprocessor = DataPreprocessor(fiscal_start_month=10)  # October start
    predictor = TimeSeriesPredictor(
        yearly_seasonal_periods=252,  # Working days in a year
        weekly_seasonal_periods=5,  # 5-day work week
    )
    simulator = MonteCarloSimulator(n_simulations=1000)
    kpi_calc = KPICalculator()

    # Load and prepare data
    raw_data = preprocessor.load_and_prepare_data("data/generated/labor_data.csv")
    daily_data = preprocessor.aggregate_daily_metrics(raw_data)
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

    # Calculate KPIs
    labor_kpis = kpi_calc.calculate_labor_efficiency(daily_data)

    # Visualize results
    plot_results(daily_data, mean_forecast, lower_bound, upper_bound)
    plot_fiscal_pattern(fiscal_pattern)

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
