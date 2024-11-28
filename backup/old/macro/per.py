import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Generate dummy data
num_employees = 20
num_days = 1460  # Four years of daily data
employee_ids = [f"Emp_{i+1}" for i in range(num_employees)]

date_range = pd.date_range(start="2022-01-01", periods=num_days, freq="D")
data = pd.DataFrame({"Date": date_range})
data = (
    data.assign(key=1)
    .merge(pd.DataFrame({"Employee_ID": employee_ids, "key": 1}), on="key")
    .drop("key", axis=1)
)

# Simulate total hours data
np.random.seed(42)
base_hours = 8
weekend_base_hours = 2


def simulate_hours(date):
    is_weekend = date.dayofweek >= 5
    daily_base = weekend_base_hours if is_weekend else base_hours
    if is_weekend:
        return np.random.uniform(0, daily_base) if np.random.rand() < 0.3 else 0
    return np.random.uniform(daily_base - 1, daily_base + 1)


data["Total_Hours"] = data["Date"].apply(simulate_hours)

# Aggregate to weekly data
data.loc[:, "Week_End"] = data["Date"] + pd.to_timedelta(
    (4 - data["Date"].dt.dayofweek) % 7, unit="D"
)
weekly_data = data.groupby("Week_End")["Total_Hours"].sum().reset_index()

# Time Series Analysis
weekly_data.set_index("Week_End", inplace=True)
weekly_data = weekly_data.asfreq("W-FRI")

train_data = weekly_data.iloc[:-4]
test_data = weekly_data.iloc[-4:]

model = ExponentialSmoothing(
    train_data["Total_Hours"], trend="add", seasonal="add", seasonal_periods=52
)
hw_fit = model.fit()
hw_forecast = hw_fit.forecast(steps=4)

# Monte Carlo Simulation
residuals = train_data["Total_Hours"] - hw_fit.fittedvalues
residual_mean, residual_std = residuals.mean(), residuals.std()

num_simulations = 1000
simulation_results = pd.DataFrame(index=test_data.index, columns=range(num_simulations))

for i in range(num_simulations):
    random_residuals = np.random.normal(residual_mean, residual_std, 4)
    simulation_results[i] = hw_forecast + random_residuals

forecast_mean = simulation_results.mean(axis=1)
confidence_interval_5 = simulation_results.quantile(0.05, axis=1)
confidence_interval_95 = simulation_results.quantile(0.95, axis=1)

# Combine historical data, forecasts, and confidence intervals
combined_data = pd.concat(
    [
        train_data,
        pd.DataFrame(
            {
                "Total_Hours": test_data["Total_Hours"],
                "Deterministic_Forecast": hw_forecast,
                "Monte_Carlo_Forecast": forecast_mean,
                "Confidence_Interval_5": confidence_interval_5,
                "Confidence_Interval_95": confidence_interval_95,
            }
        ),
    ]
)

# Reset index to make the date a column
combined_data = combined_data.reset_index()

# Rename the index column to 'Date' for clarity
combined_data = combined_data.rename(columns={"Week_End": "Date"})

# Ensure all date values are in a consistent format
combined_data["Date"] = pd.to_datetime(combined_data["Date"])

# Step 1: Count distinct employees per week
employee_weekly_count = data.groupby("Week_End")["Employee_ID"].nunique().reset_index()
employee_weekly_count.columns = ["Date", "Employee_Count"]

# Merge the employee count into the combined data
combined_data = pd.merge(combined_data, employee_weekly_count, on="Date", how="left")

# Step 2: Divide Total Hours, Forecasts, and Confidence Intervals by employee count
combined_data["Total_Hours_Per_Employee"] = (
    combined_data["Total_Hours"] / combined_data["Employee_Count"]
)
combined_data["Deterministic_Forecast_Per_Employee"] = (
    combined_data["Deterministic_Forecast"] / combined_data["Employee_Count"]
)
combined_data["Monte_Carlo_Forecast_Per_Employee"] = (
    combined_data["Monte_Carlo_Forecast"] / combined_data["Employee_Count"]
)
combined_data["Confidence_Interval_5_Per_Employee"] = (
    combined_data["Confidence_Interval_5"] / combined_data["Employee_Count"]
)
combined_data["Confidence_Interval_95_Per_Employee"] = (
    combined_data["Confidence_Interval_95"] / combined_data["Employee_Count"]
)
