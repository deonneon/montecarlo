import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple

# Constants
YEARLY_SEASONAL_PERIODS = 252  # Approximately 252 working days per year
WEEKLY_SEASONAL_PERIODS = 5  # 5-day work week


def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    """Load and prepare data from CSV file"""
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    daily_data = (
        df.groupby("date")["total_hours_charged"]
        .sum()
        .reset_index()
        .set_index("date")
        .sort_index()
    )

    return daily_data


def create_prophet_model():
    """Create a new Prophet model instance with specified parameters"""

    # Create a DataFrame with the custom holidays
    holidays_df = pd.DataFrame(
        {
            "ds": [
                # Year 2019
                datetime(2019, 1, 1),  # New Year's Day
                datetime(2019, 1, 21),  # Martin Luther King Jr. Day
                datetime(2019, 2, 18),  # Presidents' Day
                datetime(2019, 5, 27),  # Memorial Day
                datetime(2019, 7, 4),  # Independence Day
                datetime(2019, 9, 2),  # Labor Day
                datetime(2019, 10, 14),  # Columbus Day
                datetime(2019, 11, 11),  # Veterans Day
                datetime(2019, 11, 28),  # Thanksgiving Day
                datetime(2019, 12, 25),  # Christmas Day
                # Year 2020
                datetime(2020, 1, 1),  # New Year's Day
                datetime(2020, 1, 20),  # Martin Luther King Jr. Day
                datetime(2020, 2, 17),  # Presidents' Day
                datetime(2020, 5, 25),  # Memorial Day
                datetime(2020, 7, 4),  # Independence Day
                datetime(2020, 9, 7),  # Labor Day
                datetime(2020, 10, 12),  # Columbus Day
                datetime(2020, 11, 11),  # Veterans Day
                datetime(2020, 11, 26),  # Thanksgiving Day
                datetime(2020, 12, 25),  # Christmas Day
                # Year 2021
                datetime(2021, 1, 1),  # New Year's Day
                datetime(2021, 1, 18),  # Martin Luther King Jr. Day
                datetime(2021, 2, 15),  # Presidents' Day
                datetime(2021, 5, 31),  # Memorial Day
                datetime(2021, 6, 19),  # Juneteenth National Independence Day
                datetime(2021, 7, 4),  # Independence Day
                datetime(2021, 9, 6),  # Labor Day
                datetime(2021, 10, 11),  # Columbus Day
                datetime(2021, 11, 11),  # Veterans Day
                datetime(2021, 11, 25),  # Thanksgiving Day
                datetime(2021, 12, 25),  # Christmas Day
                # Year 2022
                datetime(2022, 1, 1),  # New Year's Day
                datetime(2022, 1, 17),  # Martin Luther King Jr. Day
                datetime(2022, 2, 21),  # Presidents' Day
                datetime(2022, 5, 30),  # Memorial Day
                datetime(2022, 6, 19),  # Juneteenth National Independence Day
                datetime(2022, 7, 4),  # Independence Day
                datetime(2022, 9, 5),  # Labor Day
                datetime(2022, 10, 10),  # Columbus Day
                datetime(2022, 11, 11),  # Veterans Day
                datetime(2022, 11, 24),  # Thanksgiving Day
                datetime(2022, 12, 25),  # Christmas Day
                # Year 2023
                datetime(2023, 1, 1),  # New Year's Day
                datetime(2023, 1, 16),  # Martin Luther King Jr. Day
                datetime(2023, 2, 20),  # Presidents' Day
                datetime(2023, 5, 29),  # Memorial Day
                datetime(2023, 6, 19),  # Juneteenth National Independence Day
                datetime(2023, 7, 4),  # Independence Day
                datetime(2023, 9, 4),  # Labor Day
                datetime(2023, 10, 9),  # Columbus Day
                datetime(2023, 11, 10),  # Veterans Day (observed)
                datetime(2023, 11, 23),  # Thanksgiving Day
                datetime(2023, 12, 25),  # Christmas Day
            ],
            "holiday": "US-Holiday",  # Give all holidays the same label
        }
    )

    return Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        holidays=holidays_df,
    )


def fit_prophet(data: pd.DataFrame) -> Prophet:
    """Fit Prophet model with multiple seasonality"""
    prophet_df = data.reset_index()
    prophet_df.columns = ["ds", "y"]

    model = create_prophet_model()
    model.add_seasonality(
        name="fiscal_year", period=YEARLY_SEASONAL_PERIODS, fourier_order=10
    )

    model.fit(prophet_df)
    return model


def fit_holtwinters(data: pd.DataFrame):
    """Fit Holt-Winters model"""
    model = ExponentialSmoothing(
        data["total_hours_charged"],
        seasonal_periods=365,
        trend="add",
        seasonal="add",
    ).fit()
    return model


def predict(model, forecast_horizon: int, model_type: str = "prophet"):
    """Generate predictions for specified model"""
    if model_type.lower() == "prophet":
        if model is None:
            raise ValueError("Prophet model must be fitted before making predictions")

        future = model.make_future_dataframe(
            periods=forecast_horizon, freq="D", include_history=False
        )
        forecast = model.predict(future)
        return forecast

    elif model_type.lower() == "holtwinters":
        if model is None:
            raise ValueError(
                "Holt-Winters model must be fitted before making predictions"
            )

        forecast = model.forecast(forecast_horizon)
        return forecast
    else:
        raise ValueError("Invalid model type. Choose 'prophet' or 'holtwinters'")


def evaluate_models(
    actual: np.array, prophet_forecast: np.array, hw_forecast: np.array
) -> dict:
    """Evaluate and compare model performances"""
    return {
        "Prophet": {
            "MAE": mean_absolute_error(actual, prophet_forecast),
            "RMSE": np.sqrt(mean_squared_error(actual, prophet_forecast)),
        },
        "Holt-Winters": {
            "MAE": mean_absolute_error(actual, hw_forecast),
            "RMSE": np.sqrt(mean_squared_error(actual, hw_forecast)),
        },
    }


def plot_forecasts(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    prophet_forecast: np.array,
    hw_forecast: np.array,
    title: str,
) -> None:
    """Plot the forecasts from both models"""
    plt.figure(figsize=(15, 7))

    plt.plot(
        train_data.index,
        train_data["total_hours_charged"],
        label="Training Data",
        color="black",
    )
    plt.plot(
        test_data.index,
        test_data["total_hours_charged"],
        label="Actual",
        color="blue",
    )
    plt.plot(
        test_data.index,
        prophet_forecast,
        label="Prophet Forecast",
        color="red",
        linestyle="--",
    )
    plt.plot(
        test_data.index,
        hw_forecast,
        label="Holt-Winters Forecast",
        color="green",
        linestyle="--",
    )

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Total Hours Charged")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Load data
    data = load_and_prepare_data("labor_data.csv")

    print("\nData Overview:")
    print(f"Date Range: {data.index.min()} to {data.index.max()}")
    print(f"Number of days: {len(data)}")

    # Split data
    train_size = len(data) - 180
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Fit both models
    prophet_model = fit_prophet(train_data)
    hw_model = fit_holtwinters(train_data)

    # Generate forecasts
    prophet_forecast = predict(prophet_model, 180, "prophet")
    hw_forecast = predict(hw_model, 180, "holtwinters")

    # Evaluate models
    evaluation = evaluate_models(
        test_data["total_hours_charged"].values,
        prophet_forecast["yhat"].values,
        hw_forecast.values,
    )

    # Print evaluation metrics
    print("\nModel Evaluation Metrics:")
    for model, metrics in evaluation.items():
        print(f"\n{model}:")
        print(f"MAE: {metrics['MAE']:.2f}")
        print(f"RMSE: {metrics['RMSE']:.2f}")

    # Plot results
    plot_forecasts(
        train_data,
        test_data,
        prophet_forecast["yhat"].values,
        hw_forecast.values,
        "Comparison of Forecasting Methods (180-Day Forecast)",
    )


if __name__ == "__main__":
    main()
