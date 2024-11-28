import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv("labor_data.csv", parse_dates=["date"])

# Ensure the 'userid' column is treated as a string
df["userid"] = df["userid"].astype(str)

# Extract time periods
df["year"] = df["date"].dt.year
df["quarter"] = df["date"].dt.to_period("Q")
df["month"] = df["date"].dt.to_period("M")
df["day_of_month"] = df["date"].dt.day

# Corrected line: Use .dt.start_time to access start_time of periods in a Series
df["quarter_start_date"] = df["date"].dt.to_period("Q").dt.start_time
df["day_of_quarter"] = (df["date"] - df["quarter_start_date"]).dt.days + 1

df["day_of_year"] = df["date"].dt.dayofyear

# Assuming 'today' is the last date in the dataset
today = df["date"].max()
current_year = today.year
previous_year = current_year - 1

# Define the same period last year
same_day_last_year = today - pd.DateOffset(years=1)

# MTD Comparison
# Get the day of the month
current_day_of_month = today.day

# Current Year MTD
df_mtd_current = df[
    (df["year"] == current_year) & (df["day_of_month"] <= current_day_of_month)
]
mtd_employee_count_current = df_mtd_current["userid"].nunique()

# Previous Year MTD
df_mtd_previous = df[
    (df["year"] == previous_year) & (df["day_of_month"] <= current_day_of_month)
]
mtd_employee_count_previous = df_mtd_previous["userid"].nunique()

# MTD Percentage Change
if mtd_employee_count_previous != 0:
    mtd_change_percentage = (
        (mtd_employee_count_current - mtd_employee_count_previous)
        / mtd_employee_count_previous
    ) * 100
else:
    mtd_change_percentage = float("inf")  # Handle division by zero

print(
    f"Current MTD Employee Count (as of {today.date()}): {mtd_employee_count_current}"
)
print(
    f"Previous Year MTD Employee Count (as of {same_day_last_year.date()}): {mtd_employee_count_previous}"
)
print(f"MTD Employee Count Change: {mtd_change_percentage:.2f}%")

# Add MTM comparison (after the MTD section)
# Get previous month date
previous_month = today - pd.DateOffset(months=1)

# Current Month
df_current_month = df[df["month"] == today.to_period("M")]
current_month_employees = df_current_month["userid"].nunique()

# Previous Month
df_previous_month = df[df["month"] == previous_month.to_period("M")]
previous_month_employees = df_previous_month["userid"].nunique()

# MTM Percentage Change
if previous_month_employees != 0:
    mtm_change_percentage = (
        (current_month_employees - previous_month_employees) / previous_month_employees
    ) * 100
else:
    mtm_change_percentage = float("inf")

print(
    f"\nCurrent Month Employee Count (as of {today.date()}): {current_month_employees}"
)
print(
    f"Previous Month Employee Count (as of {previous_month.date()}): {previous_month_employees}"
)
print(f"MTM Employee Count Change: {mtm_change_percentage:.2f}%")


# QTD Comparison
# Get the day of the quarter
# Corrected line: Use .to_period('Q').start_time directly as it's a Period object
current_quarter_start = today.to_period("Q").start_time
current_day_of_quarter = (today - current_quarter_start).days + 1

# Current Year QTD
df_qtd_current = df[
    (df["quarter"] == today.to_period("Q"))
    & (df["day_of_quarter"] <= current_day_of_quarter)
]
qtd_employee_count_current = df_qtd_current["userid"].nunique()

# Previous Year QTD
previous_year_quarter = (today - pd.DateOffset(years=1)).to_period("Q")
previous_quarter_start = previous_year_quarter.start_time
df_qtd_previous = df[
    (df["quarter"] == previous_year_quarter)
    & (df["day_of_quarter"] <= current_day_of_quarter)
]
qtd_employee_count_previous = df_qtd_previous["userid"].nunique()

# QTD Percentage Change
if qtd_employee_count_previous != 0:
    qtd_change_percentage = (
        (qtd_employee_count_current - qtd_employee_count_previous)
        / qtd_employee_count_previous
    ) * 100
else:
    qtd_change_percentage = float("inf")

print(
    f"\nCurrent QTD Employee Count (as of {today.date()}): {qtd_employee_count_current}"
)
print(
    f"Previous Year QTD Employee Count (as of {same_day_last_year.date()}): {qtd_employee_count_previous}"
)
print(f"QTD Employee Count Change: {qtd_change_percentage:.2f}%")

# Add QTQ comparison (after the QTD section)
# Get previous quarter date
previous_quarter = today - pd.DateOffset(months=3)

# Current Quarter
df_current_quarter = df[df["quarter"] == today.to_period("Q")]
current_quarter_employees = df_current_quarter["userid"].nunique()

# Previous Quarter
df_previous_quarter = df[df["quarter"] == previous_quarter.to_period("Q")]
previous_quarter_employees = df_previous_quarter["userid"].nunique()

# QTQ Percentage Change
if previous_quarter_employees != 0:
    qtq_change_percentage = (
        (current_quarter_employees - previous_quarter_employees)
        / previous_quarter_employees
    ) * 100
else:
    qtq_change_percentage = float("inf")

print(
    f"\nCurrent Quarter Employee Count (as of {today.date()}): {current_quarter_employees}"
)
print(
    f"Previous Quarter Employee Count (as of {previous_quarter.date()}): {previous_quarter_employees}"
)
print(f"QTQ Employee Count Change: {qtq_change_percentage:.2f}%")

# YTD Comparison
# Get the day of the year
current_day_of_year = today.timetuple().tm_yday

# Current Year YTD
df_ytd_current = df[
    (df["year"] == current_year) & (df["day_of_year"] <= current_day_of_year)
]
ytd_employee_count_current = df_ytd_current["userid"].nunique()

# Previous Year YTD
df_ytd_previous = df[
    (df["year"] == previous_year) & (df["day_of_year"] <= current_day_of_year)
]
ytd_employee_count_previous = df_ytd_previous["userid"].nunique()

# YTD Percentage Change
if ytd_employee_count_previous != 0:
    ytd_change_percentage = (
        (ytd_employee_count_current - ytd_employee_count_previous)
        / ytd_employee_count_previous
    ) * 100
else:
    ytd_change_percentage = float("inf")

print(
    f"\nCurrent YTD Employee Count (as of {today.date()}): {ytd_employee_count_current}"
)
print(
    f"Previous Year YTD Employee Count (as of {same_day_last_year.date()}): {ytd_employee_count_previous}"
)
print(f"YTD Employee Count Change: {ytd_change_percentage:.2f}%")


# Add YTY comparison (after the YTD section)
# Current Year
df_current_year = df[df["year"] == current_year]
current_year_employees = df_current_year["userid"].nunique()

# Previous Year
df_previous_year = df[df["year"] == previous_year]
previous_year_employees = df_previous_year["userid"].nunique()

# YTY Percentage Change
if previous_year_employees != 0:
    yty_change_percentage = (
        (current_year_employees - previous_year_employees) / previous_year_employees
    ) * 100
else:
    yty_change_percentage = float("inf")

print(f"\nCurrent Year Employee Count (as of {today.date()}): {current_year_employees}")
print(f"Previous Year Employee Count: {previous_year_employees}")
print(f"YTY Employee Count Change: {yty_change_percentage:.2f}%")
