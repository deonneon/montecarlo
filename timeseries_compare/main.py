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

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # Better for varying seasonality
    )
    model.add_country_holidays(country_name="US")
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

    # Split data into training and testing sets (last 90 days for testing)
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
        "Comparison of Forecasting Methods (90-Day Forecast)",
    )


if __name__ == "__main__":
    main()
