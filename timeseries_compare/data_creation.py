import pandas as pd
import random
from datetime import datetime, timedelta

# Update the date range to cover 5 years
start_date = datetime(2019, 1, 1)
end_date = datetime(2023, 12, 31)
total_days = (end_date - start_date).days + 1

# Rest of the parameters remain the same
num_employees_start = 50
num_employees_end = 70
dept_list = ["Assembly", "Quality", "Maintenance", "Logistics", "Management"]

# Extended holidays list to include 2019 and 2020
holidays = [
    # Year 2019
    datetime(2019, 1, 1),  # New Year's Day
    datetime(2019, 1, 21),  # Martin Luther King Jr. Day
    datetime(2019, 2, 18),  # Presidents' Day
    datetime(2019, 5, 27),  # Memorial Day
    datetime(2019, 7, 4),  # Independence Day
    datetime(2019, 9, 2),  # Labor Day
    datetime(2019, 10, 14),  # Columbus Day
    datetime(2019, 11, 11),  # Veterans Day
    datetime(2019, 11, 28),  # Thanksgiving Day
    datetime(2019, 12, 25),  # Christmas Day
    # Year 2020
    datetime(2020, 1, 1),  # New Year's Day
    datetime(2020, 1, 20),  # Martin Luther King Jr. Day
    datetime(2020, 2, 17),  # Presidents' Day
    datetime(2020, 5, 25),  # Memorial Day
    datetime(2020, 7, 4),  # Independence Day
    datetime(2020, 9, 7),  # Labor Day
    datetime(2020, 10, 12),  # Columbus Day
    datetime(2020, 11, 11),  # Veterans Day
    datetime(2020, 11, 26),  # Thanksgiving Day
    datetime(2020, 12, 25),  # Christmas Day
    # Year 2021
    datetime(2021, 1, 1),  # New Year's Day
    datetime(2021, 1, 18),  # Martin Luther King Jr. Day
    datetime(2021, 2, 15),  # Presidents' Day
    datetime(2021, 5, 31),  # Memorial Day
    datetime(2021, 6, 19),  # Juneteenth National Independence Day
    datetime(2021, 7, 4),  # Independence Day
    datetime(2021, 9, 6),  # Labor Day
    datetime(2021, 10, 11),  # Columbus Day
    datetime(2021, 11, 11),  # Veterans Day
    datetime(2021, 11, 25),  # Thanksgiving Day
    datetime(2021, 12, 25),  # Christmas Day
    # Year 2022
    datetime(2022, 1, 1),  # New Year's Day
    datetime(2022, 1, 17),  # Martin Luther King Jr. Day
    datetime(2022, 2, 21),  # Presidents' Day
    datetime(2022, 5, 30),  # Memorial Day
    datetime(2022, 6, 19),  # Juneteenth National Independence Day
    datetime(2022, 7, 4),  # Independence Day
    datetime(2022, 9, 5),  # Labor Day
    datetime(2022, 10, 10),  # Columbus Day
    datetime(2022, 11, 11),  # Veterans Day
    datetime(2022, 11, 24),  # Thanksgiving Day
    datetime(2022, 12, 25),  # Christmas Day
    # Year 2023
    datetime(2023, 1, 1),  # New Year's Day
    datetime(2023, 1, 16),  # Martin Luther King Jr. Day
    datetime(2023, 2, 20),  # Presidents' Day
    datetime(2023, 5, 29),  # Memorial Day
    datetime(2023, 6, 19),  # Juneteenth National Independence Day
    datetime(2023, 7, 4),  # Independence Day
    datetime(2023, 9, 4),  # Labor Day
    datetime(2023, 10, 9),  # Columbus Day
    datetime(2023, 11, 10),  # Veterans Day (observed)
    datetime(2023, 11, 23),  # Thanksgiving Day
    datetime(2023, 12, 25),  # Christmas Day
]

# The rest of the code remains the same
employees = [f"EMP{str(i).zfill(3)}" for i in range(1, num_employees_end + 1)]


# Helper functions remain unchanged
def generate_daily_labor(userid, date, dept):
    is_holiday = date in holidays
    base_hours = 0 if is_holiday else random.uniform(6, 8)
    overtime_hours = (
        0 if is_holiday else random.uniform(0, 2) if random.random() < 0.3 else 0
    )
    direct_hours = round(base_hours * random.uniform(0.6, 0.9), 2)
    non_direct_hours = round(base_hours - direct_hours, 2) if base_hours > 0 else 0
    return {
        "userid": userid,
        "dept": dept,
        "date": date,
        "total_hours_charged": round(base_hours + overtime_hours, 2),
        "direct_hours": direct_hours,
        "non_direct_hours": non_direct_hours,
        "overtime_hours": round(overtime_hours, 2),
        "is_holiday": is_holiday,
    }


# Create a growing employee base
daily_data = []
for day in range(total_days):
    current_date = start_date + timedelta(days=day)
    current_employee_count = num_employees_start + int(
        (num_employees_end - num_employees_start) * (day / total_days)
    )
    sampled_employees = random.sample(employees, current_employee_count)
    for userid in sampled_employees:
        dept = random.choice(dept_list)
        daily_data.append(generate_daily_labor(userid, current_date, dept))

# Convert the data into a DataFrame
labor_data_df = pd.DataFrame(daily_data)


def adjust_manufacturing_seasonality(df):
    df["month"] = df["date"].dt.month

    manufacturing_seasonal_factors = {
        1: 0.92,
        2: 0.96,
        3: 1.08,
        4: 1.1,
        5: 0.95,
        6: 0.92,
        7: 0.88,
        8: 0.96,
        9: 0.97,
        10: 1.15,
        11: 0.95,
        12: 0.85,
    }

    df["seasonal_factor"] = df["month"].map(manufacturing_seasonal_factors)
    df["total_hours_charged"] = round(
        df["total_hours_charged"] * df["seasonal_factor"], 2
    )
    df["direct_hours"] = round(df["direct_hours"] * df["seasonal_factor"], 2)
    df["non_direct_hours"] = round(df["non_direct_hours"] * df["seasonal_factor"], 2)
    df["overtime_hours"] = round(df["overtime_hours"] * df["seasonal_factor"], 2)

    df.drop(columns=["month", "seasonal_factor"], inplace=True)
    return df


# Apply the manufacturing-specific seasonal trends
manufacturing_labor_data_df = adjust_manufacturing_seasonality(labor_data_df)

# Save the DataFrame to a CSV file
labor_data_df.to_csv("labor_data.csv", index=False)

print(labor_data_df["date"].dt.year.unique())
