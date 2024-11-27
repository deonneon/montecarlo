# Step 1: Generate Holt-Winters Weekly Forecast (existing hybrid approach)
hw_forecast = <Holt-Winters Hybrid Code>

# Step 2: Run Monte Carlo Daily Simulations
# Simulate daily hours per employee, aggregate to weekly totals
mc_daily_results = ... # Daily Monte Carlo simulations per employee
mc_weekly_totals = mc_daily_results.groupby("Week_End")["Simulated_Hours"].sum()

# Step 3: Calculate Model Errors
hw_error = np.mean(np.abs(train_data["Total_Hours"] - hw_fit.fittedvalues))
mc_error = np.mean(np.abs(weekly_data["Total_Hours"] - mc_weekly_totals))

# Assign dynamic weights
alpha = 1 / hw_error / (1 / hw_error + 1 / mc_error)
blended_forecast = alpha * hw_forecast + (1 - alpha) * mc_weekly_totals

# Step 4: Confidence Intervals
# Combine Holt-Winters CI with Monte Carlo variability
combined_ci_5 = np.minimum(confidence_interval_5, mc_weekly_totals.quantile(0.05))
combined_ci_95 = np.maximum(confidence_interval_95, mc_weekly_totals.quantile(0.95))

# Visualization
plt.figure(figsize=(12, 6))
plt.plot(weekly_data.index, blended_forecast, label="Blended Forecast")
plt.fill_between(
    weekly_data.index, combined_ci_5, combined_ci_95, color="gray", alpha=0.3, label="Combined CI"
)
plt.legend()
plt.show()
