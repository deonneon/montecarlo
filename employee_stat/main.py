import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
df = pd.read_csv("labor_data.csv", parse_dates=["date"])

# Display the first few rows to verify the data
print(df.head())

# Total number of unique employees
total_unique_employees = df["userid"].nunique()
print(f"Total number of unique employees: {total_unique_employees}")

# Number of employees working per day
employees_per_day = df.groupby("date")["userid"].nunique().reset_index()
employees_per_day.rename(columns={"userid": "employees_working"}, inplace=True)
print(employees_per_day.head())

# Plot number of employees working per day over time
plt.figure(figsize=(12, 6))
plt.plot(employees_per_day["date"], employees_per_day["employees_working"])
plt.title("Number of Employees Working Per Day Over Time")
plt.xlabel("Date")
plt.ylabel("Number of Employees")
plt.show()

# Compute average number of employees working per month
df["year_month"] = df["date"].dt.to_period("M")
employees_per_month = df.groupby("year_month")["userid"].nunique().reset_index()
employees_per_month.rename(columns={"userid": "employees_working"}, inplace=True)
employees_per_month["year_month"] = employees_per_month["year_month"].dt.to_timestamp()

# Plot average number of employees working per month
plt.figure(figsize=(12, 6))
plt.plot(employees_per_month["year_month"], employees_per_month["employees_working"])
plt.title("Average Number of Employees Working Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Employees")
plt.show()

# Employee count per department
employees_per_dept = df.groupby("dept")["userid"].nunique().reset_index()
employees_per_dept.rename(columns={"userid": "unique_employees"}, inplace=True)
print(employees_per_dept)

# Employee count per department over time (per month)
employees_dept_month = (
    df.groupby(["year_month", "dept"])["userid"].nunique().reset_index()
)
employees_dept_month.rename(columns={"userid": "employees_working"}, inplace=True)
employees_dept_month["year_month"] = employees_dept_month[
    "year_month"
].dt.to_timestamp()

# Pivot the data to have departments as columns
dept_pivot = employees_dept_month.pivot(
    index="year_month", columns="dept", values="employees_working"
)
dept_pivot.plot(kind="line", figsize=(12, 6))
plt.title("Number of Employees Working Per Department Over Time")
plt.xlabel("Month")
plt.ylabel("Number of Employees")
plt.legend(title="Department")
plt.show()

# Average hours worked per employee
average_hours_per_employee = (
    df.groupby("userid")["total_hours_charged"].mean().reset_index()
)
average_hours_per_employee.rename(
    columns={"total_hours_charged": "average_total_hours_charged"}, inplace=True
)
print(average_hours_per_employee.head())

# Average overtime hours per employee
average_overtime_per_employee = (
    df.groupby("userid")["overtime_hours"].mean().reset_index()
)
average_overtime_per_employee.rename(
    columns={"overtime_hours": "average_overtime_hours"}, inplace=True
)
print(average_overtime_per_employee.head())

# Average direct and non-direct hours per employee
average_direct_hours = df.groupby("userid")["direct_hours"].mean().reset_index()
average_non_direct_hours = df.groupby("userid")["non_direct_hours"].mean().reset_index()
average_hours = average_direct_hours.merge(average_non_direct_hours, on="userid")
print(average_hours.head())

# Total overtime hours per month
overtime_per_month = df.groupby("year_month")["overtime_hours"].sum().reset_index()
overtime_per_month["year_month"] = overtime_per_month["year_month"].dt.to_timestamp()
plt.figure(figsize=(12, 6))
plt.plot(overtime_per_month["year_month"], overtime_per_month["overtime_hours"])
plt.title("Total Overtime Hours Per Month")
plt.xlabel("Month")
plt.ylabel("Overtime Hours")
plt.show()

# Total labor hours charged per month
total_hours_per_month = (
    df.groupby("year_month")["total_hours_charged"].sum().reset_index()
)
total_hours_per_month["year_month"] = total_hours_per_month[
    "year_month"
].dt.to_timestamp()
plt.figure(figsize=(12, 6))
plt.plot(
    total_hours_per_month["year_month"], total_hours_per_month["total_hours_charged"]
)
plt.title("Total Labor Hours Charged Per Month")
plt.xlabel("Month")
plt.ylabel("Total Hours Charged")
plt.show()

# Productivity measures
# Calculate the percentage of direct hours to total hours
df["direct_hours_percentage"] = df["direct_hours"] / df["total_hours_charged"]
average_direct_hours_percentage = (
    df.groupby("userid")["direct_hours_percentage"].mean().reset_index()
)
print(average_direct_hours_percentage.head())

# Output summary statistics
print("\nSummary Statistics:")
print(f"Total unique employees: {total_unique_employees}")
print(
    f"Average employees working per day: {employees_per_day['employees_working'].mean():.2f}"
)
print(
    f"Average employees working per month: {employees_per_month['employees_working'].mean():.2f}"
)
print(
    f"Average total hours charged per employee per day: {df.groupby('userid')['total_hours_charged'].mean().mean():.2f}"
)
print(
    f"Average overtime hours per employee per day: {df.groupby('userid')['overtime_hours'].mean().mean():.2f}"
)
print(
    f"Average direct hours percentage: {average_direct_hours_percentage['direct_hours_percentage'].mean():.2f}"
)
