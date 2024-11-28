import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # Group by date to get daily totals
    daily_data = (
        df.groupby("date")["total_hours_charged"]
        .sum()
        .reset_index()
        .set_index("date")
        .sort_index()
    )

    return daily_data


def prophet_forecast(data, periods=180):
    prophet_df = data.reset_index()
    prophet_df.columns = ["ds", "y"]

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

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # Better for varying seasonality
        holidays=holidays_df,  # Add the custom holidays
    )

    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast


def holtwinters_forecast(data, periods=180):
    model = ExponentialSmoothing(
        data["total_hours_charged"],
        seasonal_periods=365,
        trend="add",
        seasonal="add",
    ).fit()

    forecast = model.forecast(periods)
    return forecast


def evaluate_models(actual, prophet_forecast, hw_forecast):
    prophet_mae = mean_absolute_error(actual, prophet_forecast)
    prophet_rmse = np.sqrt(mean_squared_error(actual, prophet_forecast))

    hw_mae = mean_absolute_error(actual, hw_forecast)
    hw_rmse = np.sqrt(mean_squared_error(actual, hw_forecast))

    return {
        "Prophet": {"MAE": prophet_mae, "RMSE": prophet_rmse},
        "Holt-Winters": {"MAE": hw_mae, "RMSE": hw_rmse},
    }


def plot_forecasts(train_data, test_data, prophet_forecast, hw_forecast, title):
    plt.figure(figsize=(15, 7))

    plt.plot(
        train_data.index,
        train_data["total_hours_charged"],
        label="Training Data",
        color="black",
    )
    plt.plot(
        test_data.index, test_data["total_hours_charged"], label="Actual", color="blue"
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

    # Print data info for verification
    print("\nData Overview:")
    print(f"Date Range: {data.index.min()} to {data.index.max()}")
    print(f"Number of days: {len(data)}")

    # Split data into training and testing sets (last 180 days for testing)
    train_size = len(data) - 180
    train_data = data[:train_size]
    test_data = data[train_size:]

    # Generate forecasts
    prophet_pred = prophet_forecast(train_data)
    prophet_forecast_values = prophet_pred.tail(180)["yhat"].values

    hw_forecast_values = holtwinters_forecast(train_data)

    # Evaluate models
    evaluation = evaluate_models(
        test_data["total_hours_charged"].values,
        prophet_forecast_values,
        hw_forecast_values,
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
        prophet_forecast_values,
        hw_forecast_values,
        "Comparison of Forecasting Methods (180-Day Forecast)",
    )


if __name__ == "__main__":
    main()
