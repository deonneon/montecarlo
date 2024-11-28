# MSTR Cheat Sheet

## Create Metrics in MicroStrategy

- Employee Count metrics: MTD, QTD, YTD comparisons
- Hours-based metrics: total, direct, overtime hours
- Department-level metrics
- Year-over-year comparisons

### Formulation

Basic Employee Count Metrics:

Create base metrics for daily counts:

Daily Headcount = Count(DISTINCT UserID)
Department Headcount = Count(DISTINCT UserID) where Department = {Selected Dept}

Time Comparison Metrics:

Month-to-Date (MTD):

Current MTD = Count(DISTINCT UserID) where Date <= Current Day
Previous MTD = Count(DISTINCT UserID) where Date <= Same Day Last Year
MTD % Change = (Current MTD - Previous MTD) / Previous MTD

Quarter-to-Date (QTD):

Current QTD = Count(DISTINCT UserID) where Date is in Current Quarter
Previous QTD = Count(DISTINCT UserID) where Date is in Previous Year Same Quarter
QTD % Change = (Current QTD - Previous QTD) / Previous QTD

Year-to-Date (YTD):

Current YTD = Count(DISTINCT UserID) where Date is in Current Year to Date
Previous YTD = Count(DISTINCT UserID) where Date is in Previous Year Same Period
YTD % Change = (Current YTD - Previous YTD) / Previous YTD

Hours-Based Metrics:

Hours Utilization:

Total Hours = Sum(total_hours_charged)
Average Hours per Employee = Total Hours / Count(DISTINCT UserID)
Direct Hours Percentage = Sum(direct_hours) / Sum(total_hours_charged)

Overtime Analysis:

Total Overtime = Sum(overtime_hours)
Average Overtime per Employee = Total Overtime / Count(DISTINCT UserID)
Overtime Percentage = Sum(overtime_hours) / Sum(total_hours_charged)

## Employee Trends Dashboard

- Daily/Monthly headcount trends
- Department distribution
- YOY comparisons

### Formulation2

Main KPI Section:

Current headcount with YOY comparison
MTD, QTD, YTD employee metrics displayed as cards
Trend indicators showing percentage changes

Time Series Visualizations:

Line chart showing daily employee count over time
Monthly trend comparison (current year vs previous)
Department breakdown as a stacked area chart
Add drill-down capability to see detailed views

Department Analysis Section:

Bar chart showing employee distribution by department
Heat map showing department growth/decline
Department comparison table with KPIs
