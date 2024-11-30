import pandas as pd
import holidays

# List of holidays to skip
skip_holidays = {
    "Columbus Day",  # Example: skipping Columbus Day
    "Washington's Birthday",  # Example: skipping Washington's Birthday
}

# Define the years range
start_year = 2019
end_year = 2023

# Initialize U.S. federal holidays
us_holidays = holidays.US(years=range(start_year, end_year + 1))

# Generate holidays in the required format for Prophet
prophet_holidays = [
    {"ds": date, "holiday": name}
    for date, name in sorted(us_holidays.items())
    if name not in skip_holidays
]

# Create a DataFrame
holidays_df = pd.DataFrame(prophet_holidays)

# Save or display
print(holidays_df)
