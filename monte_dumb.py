import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Calculate statistics from historical data
historical_mean = weekly_data["Total_Hours"].mean()
historical_std = weekly_data["Total_Hours"].std()

# Set up Monte Carlo simulation
num_simulations = 1000
num_weeks_to_forecast = 4
simulation_results = pd.DataFrame(
    index=pd.date_range(
        start=weekly_data.index[-1] + pd.Timedelta(days=7),
        periods=num_weeks_to_forecast,
        freq="W-FRI",
    ),
    columns=range(num_simulations),
)

# Run Monte Carlo simulations
for i in range(num_simulations):
    simulation_results[i] = np.random.normal(
        historical_mean, historical_std, num_weeks_to_forecast
    )

# Calculate statistics from simulations
forecast_mean = simulation_results.mean(axis=1)
confidence_interval_5 = simulation_results.quantile(0.05, axis=1)
confidence_interval_95 = simulation_results.quantile(0.95, axis=1)

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(weekly_data.index, weekly_data["Total_Hours"], label="Historical Total Hours")
plt.plot(
    simulation_results.index,
    forecast_mean,
    label="Monte Carlo Forecast Mean",
    linestyle="--",
)
plt.fill_between(
    simulation_results.index,
    confidence_interval_5,
    confidence_interval_95,
    color="gray",
    alpha=0.3,
    label="90% Confidence Interval",
)
plt.title("Total Hours Forecast with Monte Carlo Simulation (No Time Series Model)")
plt.xlabel("Week Ending Date")
plt.ylabel("Total Hours")
plt.legend()
plt.show()
