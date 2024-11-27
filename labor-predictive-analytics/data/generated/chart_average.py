import matplotlib.pyplot as plt
import pandas as pd

manufacturing_labor_data_df = pd.read_csv("data/generated/labor_data.csv")

# Aggregate total hours per day
daily_hours = manufacturing_labor_data_df.groupby("date")["total_hours_charged"].sum()

# Calculate the number of employees working each day
daily_employee_count = manufacturing_labor_data_df.groupby("date")["userid"].nunique()

# Calculate average hours per employee per day
average_hours_per_employee = (
    manufacturing_labor_data_df.groupby("date")["total_hours_charged"].sum()
    / daily_employee_count
)

# Plot average hours per employee per day
plt.figure(figsize=(12, 6))
plt.plot(
    average_hours_per_employee.index,
    average_hours_per_employee.values,
    label="Average Hours per Employee",
)
plt.title("Average Labor Hours Charged per Employee per Day in 2021 to 2023")
plt.xlabel("Date")
plt.ylabel("Average Hours per Employee")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
