import matplotlib.pyplot as plt
import pandas as pd

manufacturing_labor_data_df = pd.read_csv("/data/generated/labor_data.csv")

# Aggregate total hours per day
daily_hours = manufacturing_labor_data_df.groupby("date")["total_hours_charged"].sum()

# Plot total hours per day
plt.figure(figsize=(12, 6))
plt.plot(daily_hours.index, daily_hours.values, label="Total Hours Charged")
plt.title("Total Labor Hours Charged per Day in 2021 to 2023")
plt.xlabel("Date")
plt.ylabel("Total Hours Charged")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
