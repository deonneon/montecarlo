import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

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

# Set random seed for reproducibility
np.random.seed(42)

# Simulate base hours
base_hours = 40

# Initialize lists to store simulated data
direct_hours = []
indirect_hours = []
leave_hours = []

for _, row in data.iterrows():
    week = row["Week_Start_Date"]
    employee_id = row["Employee_ID"]

    # Simulate seasonal leave patterns (more leave in summer and December)
    month = week.month
    if month in [6, 7, 8, 12]:
        leave_prob = 0.2  # Higher probability of taking leave
    else:
        leave_prob = 0.05  # Lower probability

    # Randomly decide if the employee takes leave this week
    is_on_leave = np.random.rand() < leave_prob

    # Simulate leave hours
    if is_on_leave:
        leave = np.random.uniform(8, 40)  # Between 1 to 5 days off
    else:
        leave = 0

    # Simulate indirect hours (e.g., training, overhead)
    indirect = np.random.uniform(2, 6)  # Between 2 to 6 hours per week

    # Calculate direct hours
    direct = max(base_hours - leave - indirect, 0)

    # Append to lists
    leave_hours.append(leave)
    indirect_hours.append(indirect)
    direct_hours.append(direct)

# Add simulated data to the DataFrame
data["Direct_Hours"] = direct_hours
data["Indirect_Hours"] = indirect_hours
data["Leave_Hours"] = leave_hours

# Ensure total hours do not exceed base hours
data["Total_Hours"] = (
    data["Direct_Hours"] + data["Indirect_Hours"] + data["Leave_Hours"]
)


######## 2. Exploratory Data Analysis

# Aggregate data by week
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

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Set the index to Week_Start_Date for time series analysis
weekly_data.set_index("Week_Start_Date", inplace=True)
weekly_data = weekly_data.asfreq("W-MON")

# Split data into training and test sets
train_data = weekly_data.iloc[:-4]  # All weeks except the last 4
test_data = weekly_data.iloc[-4:]  # Last 4 weeks to compare later

# Fit the exponential smoothing model
model = ExponentialSmoothing(
    train_data["Total_Hours"], trend="add", seasonal="add", seasonal_periods=52
)
hw_fit = model.fit()

hw_forecast = hw_fit.forecast(steps=4)

######## 4. Residual Analysis and Monte Carlo Simulation

# Calculate residuals from the training data
train_data["HW_Fitted"] = hw_fit.fittedvalues
train_data["Residuals"] = train_data["Total_Hours"] - train_data["HW_Fitted"]

# Fit a normal distribution to the residuals
residual_mean = train_data["Residuals"].mean()
residual_std = train_data["Residuals"].std()

####### Monte Carlo Simulation

# Number of simulations
num_simulations = 1000

# Initialize DataFrame to store simulation results
simulation_results = pd.DataFrame()

for i in range(num_simulations):
    # Generate random residuals
    random_residuals = np.random.normal(residual_mean, residual_std, 4)

    # Add residuals to the forecast
    simulated_forecast = hw_forecast + random_residuals

    # Store the simulation results
    simulation_results[i] = simulated_forecast.values

# Calculate mean and confidence intervals
forecast_mean = simulation_results.mean(axis=1)
forecast_std = simulation_results.std(axis=1)
confidence_interval_5 = simulation_results.quantile(0.05, axis=1)
confidence_interval_95 = simulation_results.quantile(0.95, axis=1)

######## 5. Visualization of Forecast

# Prepare dates for the forecast
forecast_dates = test_data.index

plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data["Total_Hours"], label="Historical Total Hours")
plt.plot(forecast_dates, hw_forecast, label="Deterministic Forecast", linestyle="--")
plt.plot(
    forecast_dates, forecast_mean, label="Monte Carlo Forecast Mean", linestyle="--"
)

# Plot confidence intervals
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

# Initialize dictionaries to store models and forecasts
category_models = {}
category_forecasts = {}
category_simulations = {}

categories = ["Direct_Hours", "Indirect_Hours", "Leave_Hours"]

for category in categories:
    # Fit the exponential smoothing model
    model = ExponentialSmoothing(
        train_data[category], trend="add", seasonal="add", seasonal_periods=52
    )
    fit = model.fit()

    # Forecast for the next 4 weeks
    forecast = fit.forecast(steps=4)

    # Calculate residuals
    train_data.loc[:, f"{category}_Fitted"] = fit.fittedvalues
    train_data.loc[:, f"{category}_Residuals"] = (
        train_data.loc[:, category] - fit.fittedvalues
    )

    # Fit a normal distribution to residuals
    residual_mean = train_data[f"{category}_Residuals"].mean()
    residual_std = train_data[f"{category}_Residuals"].std()

    # Monte Carlo Simulation
    simulations = pd.DataFrame()
    for i in range(num_simulations):
        random_residuals = np.random.normal(residual_mean, residual_std, 4)
        simulated_forecast = forecast + random_residuals
        simulations[i] = simulated_forecast.values

    # Store the results
    category_models[category] = fit
    category_forecasts[category] = forecast
    category_simulations[category] = simulations


###### Aggregate Simulated Forecasts

# Sum the simulations across categories to get total hours
total_simulations = (
    category_simulations["Direct_Hours"]
    + category_simulations["Indirect_Hours"]
    + category_simulations["Leave_Hours"]
)

# Calculate mean and confidence intervals
total_forecast_mean = total_simulations.mean(axis=1)
total_forecast_std = total_simulations.std(axis=1)
total_confidence_interval_5 = total_simulations.quantile(0.05, axis=1)
total_confidence_interval_95 = total_simulations.quantile(0.95, axis=1)

## plot

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
