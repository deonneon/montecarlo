import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ... [Previous data generation code remains the same] ...

# Time Series Analysis
weekly_data.set_index("Week_End", inplace=True)
weekly_data = weekly_data.asfreq("W-FRI")

train_data = weekly_data.iloc[:-4]
test_data = weekly_data.iloc[-4:]

# Define multiple seasonal periods
seasonal_periods = [52, 13, 4]  # Annual, Quarterly, Monthly

# Create SARIMAX model with multiple seasonal periods
order = (1, 1, 1)  # (p, d, q) for the non-seasonal part
seasonal_orders = [
    (1, 1, 1, s) for s in seasonal_periods
]  # (P, D, Q, s) for each seasonal component

model = SARIMAX(
    train_data["Total_Hours"],
    order=order,
    seasonal_order=seasonal_orders,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

results = model.fit()

# Forecast
forecast_steps = 4
forecast = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Monte Carlo Simulation
residuals = train_data["Total_Hours"] - results.fittedvalues
residual_mean, residual_std = residuals.mean(), residuals.std()

num_simulations = 1000
simulation_results = pd.DataFrame(index=test_data.index, columns=range(num_simulations))

for i in range(num_simulations):
    random_residuals = np.random.normal(residual_mean, residual_std, forecast_steps)
    simulation_results[i] = forecast_mean + random_residuals

sim_mean = simulation_results.mean(axis=1)
sim_ci_5 = simulation_results.quantile(0.05, axis=1)
sim_ci_95 = simulation_results.quantile(0.95, axis=1)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(train_data.index, train_data["Total_Hours"], label="Historical Total Hours")
plt.plot(test_data.index, forecast_mean, label="SARIMAX Forecast", linestyle="--")
plt.plot(test_data.index, sim_mean, label="Monte Carlo Mean", linestyle=":")
plt.fill_between(
    test_data.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="red",
    alpha=0.2,
    label="SARIMAX 95% CI",
)
plt.fill_between(
    test_data.index,
    sim_ci_5,
    sim_ci_95,
    color="gray",
    alpha=0.2,
    label="Monte Carlo 90% CI",
)
plt.title("Total Hours Forecast with Multiple Seasonal Periods")
plt.xlabel("Week Ending Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()
