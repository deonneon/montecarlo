import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

######## 1. Generate Dummy Data
num_employees = 20
num_weeks = 208  # Four years of data
employee_ids = [f"Emp_{i+1}" for i in range(num_employees)]

# Create a DataFrame to hold the time charging data
date_range = pd.date_range(start="2022-01-03", periods=num_weeks, freq="W-MON")
data = pd.DataFrame({"Week_Start_Date": date_range})

# Expand the DataFrame to have one row per employee per week
data = (
    data.assign(key=1)
    .merge(pd.DataFrame({"Employee_ID": employee_ids, "key": 1}), on="key")
    .drop("key", axis=1)
)

######## Simulate Charging Data
np.random.seed(42)
base_hours = 40
direct_hours, indirect_hours, leave_hours = [], [], []

for _, row in data.iterrows():
    week = row["Week_Start_Date"]
    employee_id = row["Employee_ID"]
    month = week.month

    # Simulate leave hours
    leave_prob = 0.2 if month in [6, 7, 8, 12] else 0.05
    is_on_leave = np.random.rand() < leave_prob
    leave = np.random.uniform(8, 40) if is_on_leave else 0

    # Simulate indirect hours and calculate direct hours
    indirect = np.random.uniform(2, 6)
    direct = max(base_hours - leave - indirect, 0)

    leave_hours.append(leave)
    indirect_hours.append(indirect)
    direct_hours.append(direct)

data["Direct_Hours"] = direct_hours
data["Indirect_Hours"] = indirect_hours
data["Leave_Hours"] = leave_hours
data["Total_Hours"] = (
    data["Direct_Hours"] + data["Indirect_Hours"] + data["Leave_Hours"]
)

######## 2. Exploratory Data Analysis
weekly_data = data.groupby("Week_Start_Date").sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(
    weekly_data["Week_Start_Date"], weekly_data["Total_Hours"], label="Total Hours"
)
plt.plot(
    weekly_data["Week_Start_Date"], weekly_data["Leave_Hours"], label="Leave Hours"
)
plt.plot(
    weekly_data["Week_Start_Date"],
    weekly_data["Indirect_Hours"],
    label="Indirect Hours",
)
plt.plot(
    weekly_data["Week_Start_Date"], weekly_data["Direct_Hours"], label="Direct Hours"
)
plt.title("Weekly Aggregated Time Charging Data")
plt.xlabel("Week Start Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()

######## 3. Time Series Analysis
weekly_data.set_index("Week_Start_Date", inplace=True)
weekly_data = weekly_data.asfreq("W-MON")

train_data = weekly_data.iloc[:-4]
test_data = weekly_data.iloc[-4:]

model = ExponentialSmoothing(
    train_data["Total_Hours"], trend="add", seasonal="add", seasonal_periods=52
)
hw_fit = model.fit()
hw_forecast = hw_fit.forecast(steps=4)

######## 4. Residual Analysis and Monte Carlo Simulation
train_data["HW_Fitted"] = hw_fit.fittedvalues
train_data["Residuals"] = train_data["Total_Hours"] - hw_fit.fittedvalues

residual_mean = train_data["Residuals"].mean()
residual_std = train_data["Residuals"].std()

num_simulations = 1000
simulation_results = pd.DataFrame()

for i in range(num_simulations):
    random_residuals = np.random.normal(residual_mean, residual_std, 4)
    simulated_forecast = hw_forecast + random_residuals
    simulation_results[i] = simulated_forecast.values

forecast_mean = simulation_results.mean(axis=1)
forecast_std = simulation_results.std(axis=1)
confidence_interval_5 = simulation_results.quantile(0.05, axis=1)
confidence_interval_95 = simulation_results.quantile(0.95, axis=1)

######## 5. Visualization of Forecast
forecast_dates = test_data.index

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data["Total_Hours"], label="Historical Total Hours")
plt.plot(forecast_dates, hw_forecast, label="Deterministic Forecast", linestyle="--")
plt.plot(
    forecast_dates, forecast_mean, label="Monte Carlo Forecast Mean", linestyle="--"
)
plt.fill_between(
    forecast_dates,
    confidence_interval_5,
    confidence_interval_95,
    color="gray",
    alpha=0.3,
    label="90% Confidence Interval",
)
plt.title("Total Hours Forecast with Monte Carlo Simulation")
plt.xlabel("Week Start Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()

######## 6. Breakdown by Categories
category_models = {}
category_forecasts = {}
category_simulations = {}

categories = ["Direct_Hours", "Indirect_Hours", "Leave_Hours"]

for category in categories:
    model = ExponentialSmoothing(
        train_data[category], trend="add", seasonal="add", seasonal_periods=52
    )
    fit = model.fit()

    forecast = fit.forecast(steps=4)

    train_data[f"{category}_Fitted"] = fit.fittedvalues
    train_data[f"{category}_Residuals"] = train_data.loc[:, category] - fit.fittedvalues

    residual_mean = train_data[f"{category}_Residuals"].mean()
    residual_std = train_data[f"{category}_Residuals"].std()

    simulations = pd.DataFrame()

    for i in range(num_simulations):
        random_residuals = np.random.normal(residual_mean, residual_std, 4)
        simulated_forecast = forecast + random_residuals
        simulations[i] = simulated_forecast.values

    category_models[category] = fit
    category_forecasts[category] = forecast
    category_simulations[category] = simulations

# Sum the simulations across categories
total_simulations = (
    category_simulations["Direct_Hours"]
    + category_simulations["Indirect_Hours"]
    + category_simulations["Leave_Hours"]
)

total_forecast_mean = total_simulations.mean(axis=1)
total_confidence_interval_5 = total_simulations.quantile(0.05, axis=1)
total_confidence_interval_95 = total_simulations.quantile(0.95, axis=1)

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data["Direct_Hours"], label="Historical Direct Hours")
plt.plot(
    forecast_dates,
    category_forecasts["Direct_Hours"],
    label="Direct Hours Forecast",
    linestyle="--",
)
plt.plot(
    train_data.index, train_data["Indirect_Hours"], label="Historical Indirect Hours"
)
plt.plot(
    forecast_dates,
    category_forecasts["Indirect_Hours"],
    label="Indirect Hours Forecast",
    linestyle="--",
)
plt.plot(train_data.index, train_data["Leave_Hours"], label="Historical Leave Hours")
plt.plot(
    forecast_dates,
    category_forecasts["Leave_Hours"],
    label="Leave Hours Forecast",
    linestyle="--",
)
plt.title("Forecast by Category with Monte Carlo Simulation")
plt.xlabel("Week Start Date")
plt.ylabel("Hours")
plt.legend()
plt.axvline(x=train_data.index[-1], color="red", linestyle="--", label="Forecast Start")
plt.show()
