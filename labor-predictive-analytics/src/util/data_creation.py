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

# Updated holidays DataFrame (Prophet-compatible format)
holidays_df = pd.DataFrame(
    {
        "ds": [
            "2019-01-01",
            "2019-01-21",
            "2019-02-18",
            "2019-05-27",
            "2019-07-04",
            "2019-09-02",
            "2019-10-14",
            "2019-11-11",
            "2019-11-28",
            "2019-12-25",
            "2020-01-01",
            "2020-01-20",
            "2020-02-17",
            "2020-05-25",
            "2020-07-04",
            "2020-09-07",
            "2020-10-12",
            "2020-11-11",
            "2020-11-26",
            "2020-12-25",
            "2021-01-01",
            "2021-01-18",
            "2021-02-15",
            "2021-05-31",
            "2021-06-19",
            "2021-07-04",
            "2021-09-06",
            "2021-10-11",
            "2021-11-11",
            "2021-11-25",
            "2021-12-25",
            "2022-01-01",
            "2022-01-17",
            "2022-02-21",
            "2022-05-30",
            "2022-06-19",
            "2022-07-04",
            "2022-09-05",
            "2022-10-10",
            "2022-11-11",
            "2022-11-24",
            "2022-12-25",
            "2023-01-01",
            "2023-01-16",
            "2023-02-20",
            "2023-05-29",
            "2023-06-19",
            "2023-07-04",
            "2023-09-04",
            "2023-10-09",
            "2023-11-10",
            "2023-11-23",
            "2023-12-25",
        ],
        "holiday": [
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Memorial Day",
            "Independence Day",
            "Labor Day",
            "Columbus Day",
            "Veterans Day",
            "Thanksgiving Day",
            "Christmas Day",
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Memorial Day",
            "Independence Day",
            "Labor Day",
            "Columbus Day",
            "Veterans Day",
            "Thanksgiving Day",
            "Christmas Day",
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Memorial Day",
            "Juneteenth",
            "Independence Day",
            "Labor Day",
            "Columbus Day",
            "Veterans Day",
            "Thanksgiving Day",
            "Christmas Day",
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Memorial Day",
            "Juneteenth",
            "Independence Day",
            "Labor Day",
            "Columbus Day",
            "Veterans Day",
            "Thanksgiving Day",
            "Christmas Day",
            "New Year's Day",
            "Martin Luther King Jr. Day",
            "Presidents' Day",
            "Memorial Day",
            "Juneteenth",
            "Independence Day",
            "Labor Day",
            "Columbus Day",
            "Veterans Day (Observed)",
            "Thanksgiving Day",
            "Christmas Day",
        ],
    }
)
holidays_df["ds"] = pd.to_datetime(holidays_df["ds"])  # Convert to datetime


# Function to generate daily labor
def generate_daily_labor(userid, date, dept):
    is_holiday = date in holidays_df["ds"].values
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


# Initialize the active employees list
active_employees = []
for i in range(1, num_employees_start + 1):
    userid = f"EMP{str(i).zfill(3)}"
    dept = random.choice(dept_list)
    active_employees.append((userid, dept))  # Tuple of (userid, department)

# Create data with growing employee base
daily_data = []
for day in range(total_days):
    current_date = start_date + timedelta(days=day)
    current_employee_count = num_employees_start + int(
        (num_employees_end - num_employees_start) * (day / total_days)
    )

    # Add new employees if needed
    while len(active_employees) < current_employee_count:
        new_emp_num = len(active_employees) + 1
        userid = f"EMP{str(new_emp_num).zfill(3)}"
        dept = random.choice(dept_list)
        active_employees.append((userid, dept))

    # Use all active employees for that day
    for userid, dept in active_employees:
        daily_data.append(generate_daily_labor(userid, current_date, dept))

# Convert the data into a DataFrame
labor_data_df = pd.DataFrame(daily_data)


# Apply manufacturing seasonality adjustments
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


manufacturing_labor_data_df = adjust_manufacturing_seasonality(labor_data_df)

# Save the DataFrame to a CSV file
manufacturing_labor_data_df.to_csv("labor_data.csv", index=False)
