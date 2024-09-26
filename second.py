import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

######## 1. Generate Dummy Data
num_employees = 20
num_days = 1460  # Four years of daily data
employee_ids = [f"Emp_{i+1}" for i in range(num_employees)]

# Create a DataFrame to hold the time charging data
date_range = pd.date_range(start="2022-01-01", periods=num_days, freq="D")
data = pd.DataFrame({"Date": date_range})

# Expand the DataFrame to have one row per employee per day
data = (
    data.assign(key=1)
    .merge(pd.DataFrame({"Employee_ID": employee_ids, "key": 1}), on="key")
    .drop("key", axis=1)
)

######## Simulate Charging Data
np.random.seed(42)
base_hours = 8  # Daily base hours for weekdays
weekend_base_hours = 2  # Base hours for weekends
direct_hours, indirect_hours = [], []

for _, row in data.iterrows():
    date = row["Date"]
    employee_id = row["Employee_ID"]
    day_of_week = date.dayofweek

    # Determine if it's a weekend
    is_weekend = day_of_week >= 5

    # Set base hours based on weekday/weekend
    daily_base = weekend_base_hours if is_weekend else base_hours

    # Simulate hours (reduced on weekends)
    if is_weekend:
        # Some weekends might have zero hours, others might have some work
        weekend_work_prob = 0.3
        if np.random.rand() < weekend_work_prob:
            direct = np.random.uniform(0, daily_base * 0.8)
            indirect = np.random.uniform(0, daily_base * 0.2)
        else:
            direct = indirect = 0
    else:
        # Weekday hours
        indirect = np.random.uniform(0.5, 1.5)
        direct = daily_base - indirect

    direct_hours.append(direct)
    indirect_hours.append(indirect)

data["Direct_Hours"] = direct_hours
data["Indirect_Hours"] = indirect_hours
data["Total_Hours"] = data["Direct_Hours"] + data["Indirect_Hours"]

######## 2. Aggregate to Weekly Data
# Calculate days until Friday (where Friday is considered 4)
days_to_friday = (4 - data["Date"].dt.dayofweek) % 7

# Add the calculated days to get to Friday
data["Week_End"] = data["Date"] + pd.to_timedelta(days_to_friday, unit="D")

# Group by Week_End and Employee_ID, sum only the numeric columns
numeric_columns = ["Direct_Hours", "Indirect_Hours", "Total_Hours"]
weekly_data = (
    data.groupby(["Week_End", "Employee_ID"])[numeric_columns].sum().reset_index()
)

# Now group by Week_End to get the total across all employees
weekly_data = weekly_data.groupby("Week_End")[numeric_columns].sum().reset_index()

######## 3. Exploratory Data Analysis
plt.figure(figsize=(12, 6))
plt.plot(weekly_data["Week_End"], weekly_data["Total_Hours"], label="Total Hours")
plt.plot(weekly_data["Week_End"], weekly_data["Indirect_Hours"], label="Indirect Hours")
plt.plot(weekly_data["Week_End"], weekly_data["Direct_Hours"], label="Direct Hours")
plt.title("Weekly Aggregated Time Charging Data")
plt.xlabel("Week Ending Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()

######## 4. Time Series Analysis
weekly_data.set_index("Week_End", inplace=True)
weekly_data = weekly_data.asfreq("W-FRI")

train_data = weekly_data.iloc[:-4]  # Use last 4 weeks for testing
test_data = weekly_data.iloc[-4:]

model = ExponentialSmoothing(
    train_data["Total_Hours"], trend="add", seasonal="add", seasonal_periods=52
)
hw_fit = model.fit()
hw_forecast = hw_fit.forecast(steps=4)

######## 5. Residual Analysis and Monte Carlo Simulation
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

######## 6. Visualization of Forecast
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
plt.xlabel("Week Ending Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()

######## 7. Breakdown by Categories
category_models = {}
category_forecasts = {}
category_simulations = {}

categories = ["Direct_Hours", "Indirect_Hours"]

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
    category_simulations["Direct_Hours"] + category_simulations["Indirect_Hours"]
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
plt.title("Forecast by Category with Monte Carlo Simulation")
plt.xlabel("Week Ending Date")
plt.ylabel("Hours")
plt.legend()
plt.axvline(x=train_data.index[-1], color="red", linestyle="--", label="Forecast Start")
plt.show()
