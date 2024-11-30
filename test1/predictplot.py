import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style using seaborn default style
sns.set_theme()
sns.set_style("whitegrid")

# Read the original labor data
labor_data = pd.read_csv("labor_data.csv")
labor_data["date"] = pd.to_datetime(labor_data["date"])

# Aggregate daily total hours across all employees
daily_totals = labor_data.groupby("date")["total_hours_charged"].sum().reset_index()

# Read the forecast data files
prophet_data = pd.read_csv("prophet_forecast_data.csv")
prophet_agg = pd.read_csv("employee_prophet_aggregated.csv")
monte_carlo = pd.read_csv("monte_carlo_results.csv")

# Convert date columns to datetime
prophet_data["date"] = pd.to_datetime(prophet_data["date"])
prophet_agg["ds"] = pd.to_datetime(prophet_agg["ds"])
monte_carlo["date"] = pd.to_datetime(monte_carlo["date"])

# Create the plot
plt.figure(figsize=(15, 8))

# Plot actual data from the original labor data
plt.plot(
    daily_totals["date"],
    daily_totals["total_hours_charged"],
    label="Actual Hours",
    color="black",
    linewidth=2,
)

# Plot general Prophet forecast
plt.plot(
    prophet_data["date"],
    prophet_data["forecast"],
    label="General Prophet Forecast",
    linestyle="--",
)
plt.fill_between(
    prophet_data["date"],
    prophet_data["forecast_lower"],
    prophet_data["forecast_upper"],
    alpha=0.2,
)

# Plot aggregated employee-level Prophet forecast
plt.plot(
    prophet_agg["ds"],
    prophet_agg["yhat"],
    label="Aggregated Employee Prophet Forecast",
    linestyle="-.",
)
plt.fill_between(
    prophet_agg["ds"], prophet_agg["yhat_lower"], prophet_agg["yhat_upper"], alpha=0.2
)

# Plot Monte Carlo simulation results
plt.plot(
    monte_carlo["date"], monte_carlo["mc_mean"], label="Monte Carlo Mean", linestyle=":"
)
plt.fill_between(
    monte_carlo["date"], monte_carlo["mc_lower"], monte_carlo["mc_upper"], alpha=0.2
)

# Customize the plot
plt.title("Labor Hours Forecast Comparison (2019-2023)", fontsize=14, pad=20)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Hours", fontsize=12)
plt.legend(loc="best", fontsize=10)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig("forecast_comparison_all.png", dpi=300, bbox_inches="tight")
plt.close()
