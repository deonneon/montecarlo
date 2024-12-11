import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate

# Initialize the Dash app with Bootstrap
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)


def create_kpi_tooltip(metric_name):
    tooltips = {
        "Utilization Rate": "Direct hours as a percentage of total hours charged",
        "Employee Count": "Average number of active employees during the period",
        "Overtime %": "Overtime hours as a percentage of total hours charged",
        "Efficiency Score": "Average hours worked per employee per day",
    }
    return tooltips[metric_name]


# Load the data
def load_data():
    monthly_kpis = pd.read_csv("monthly_kpis.csv")
    weekly_data = pd.read_csv("weekly_aggregated_data.csv")
    prophet_forecast = pd.read_csv("prophet_forecast_data.csv")
    monte_carlo = pd.read_csv("monte_carlo_results.csv")
    efficiency_kpis = pd.read_csv("efficiency_kpis.csv")
    dept_performance = pd.read_csv("dept_performance.csv")

    # Convert date columns
    date_columns = ["date", "ds"]
    for df in [weekly_data, prophet_forecast, monte_carlo]:
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

    return (
        monthly_kpis,
        weekly_data,
        prophet_forecast,
        monte_carlo,
        efficiency_kpis,
        dept_performance,
    )


# Overview Tab Layout
def create_overview_layout():
    return html.Div(
        [
            # KPI Cards
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Utilization Rate",
                                                        className="card-title",
                                                        id="utilization-rate-title",
                                                    ),
                                                    dbc.Tooltip(
                                                        create_kpi_tooltip(
                                                            "Utilization Rate"
                                                        ),
                                                        target="utilization-rate-title",
                                                    ),
                                                    html.H2(
                                                        id="utilization-rate",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="utilization-trend-indicator",
                                                        className="trend-indicator",
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="kpi-card",
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Employee Count",
                                                        className="card-title",
                                                        id="employee-count-title",
                                                    ),
                                                    dbc.Tooltip(
                                                        create_kpi_tooltip(
                                                            "Employee Count"
                                                        ),
                                                        target="employee-count-title",
                                                    ),
                                                    html.H2(
                                                        id="employee-count",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="headcount-trend-indicator",
                                                        className="trend-indicator",
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="kpi-card",
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Overtime %",
                                                        className="card-title",
                                                        id="overtime-title",
                                                    ),
                                                    dbc.Tooltip(
                                                        create_kpi_tooltip(
                                                            "Overtime %"
                                                        ),
                                                        target="overtime-title",
                                                    ),
                                                    html.H2(
                                                        id="overtime-percentage",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="overtime-trend-indicator",
                                                        className="trend-indicator",
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="kpi-card",
                                    )
                                ],
                                width=3,
                            ),
                            dbc.Col(
                                [
                                    dbc.Card(
                                        [
                                            dbc.CardBody(
                                                [
                                                    html.H4(
                                                        "Efficiency Score",
                                                        className="card-title",
                                                        id="efficiency-title",
                                                    ),
                                                    dbc.Tooltip(
                                                        create_kpi_tooltip(
                                                            "Efficiency Score"
                                                        ),
                                                        target="efficiency-title",
                                                    ),
                                                    html.H2(
                                                        id="efficiency-score",
                                                        className="kpi-value",
                                                    ),
                                                    html.P(
                                                        id="efficiency-trend-indicator",
                                                        className="trend-indicator",
                                                    ),
                                                ]
                                            )
                                        ],
                                        className="kpi-card",
                                    )
                                ],
                                width=3,
                            ),
                        ],
                        className="mb-4",
                    )
                ],
                className="kpi-container",
            ),
            # Filters
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Department"),
                                    dcc.Dropdown(
                                        id="dept-filter",
                                        options=[
                                            {
                                                "label": "All Departments",
                                                "value": "all",
                                            },
                                            {"label": "Assembly", "value": "Assembly"},
                                            {"label": "Quality", "value": "Quality"},
                                            {
                                                "label": "Maintenance",
                                                "value": "Maintenance",
                                            },
                                            {
                                                "label": "Logistics",
                                                "value": "Logistics",
                                            },
                                            {
                                                "label": "Management",
                                                "value": "Management",
                                            },
                                        ],
                                        value="all",
                                    ),
                                ],
                                width=4,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Date Range"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        start_date=datetime(2023, 1, 1),
                                        end_date=datetime(2023, 12, 31),
                                    ),
                                ],
                                width=4,
                            ),
                        ],
                        className="mb-4",
                    )
                ]
            ),
            # Charts
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    dcc.Graph(id="utilization-trend"),
                                    dcc.Graph(id="productivity-metrics"),
                                ],
                                width=6,
                            ),
                            dbc.Col(
                                [
                                    dcc.Graph(id="department-comparison"),
                                    dcc.Graph(id="efficiency-metrics"),
                                ],
                                width=6,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [dbc.Col([dcc.Graph(id="holiday-impact-analysis")], width=12)]
                    ),
                ]
            ),
        ]
    )


# Department Analytics Tab Layout
def create_department_layout():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Department Metrics"),
                            dcc.Dropdown(
                                id="dept-metrics-selector",
                                options=[
                                    {
                                        "label": "Utilization Rate",
                                        "value": "utilization",
                                    },
                                    {"label": "Overtime Hours", "value": "overtime"},
                                    {
                                        "label": "Direct Labor Hours",
                                        "value": "direct_labor",
                                    },
                                ],
                                value="utilization",
                            ),
                            dcc.Graph(id="dept-metrics-chart"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Department Comparison"),
                            dcc.Graph(id="dept-comparison-chart"),
                        ],
                        width=6,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Department Performance Trends"),
                            dcc.Graph(id="dept-performance-trend"),
                        ],
                        width=12,
                    )
                ]
            ),
        ]
    )


app.layout = html.Div(
    [
        # Header with navigation
        html.Div(
            [
                html.H1(
                    "Labor KPI Dashboard",
                    style={
                        "textAlign": "center",
                        "color": "#2c3e50",
                        "marginBottom": 20,
                    },
                ),
                dcc.Tabs(
                    id="tabs",
                    value="overview",
                    children=[
                        dcc.Tab(label="Overview", value="overview"),
                        dcc.Tab(label="Department Analytics", value="department"),
                        dcc.Tab(label="Forecasting", value="forecasting"),
                        dcc.Tab(label="Employee Metrics", value="employee"),
                    ],
                ),
            ]
        ),
        # Content div
        html.Div(id="tab-content"),
    ],
    style={"padding": "20px"},
)


# Callbacks
@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_content(tab):
    if tab == "overview":
        return create_overview_layout()
    elif tab == "department":
        return create_department_layout()
    elif tab == "forecasting":
        return create_forecasting_layout()
    elif tab == "employee":
        return create_employee_layout()


@app.callback(
    [
        Output("utilization-rate", "children"),
        Output("employee-count", "children"),
        Output("overtime-percentage", "children"),
        Output("efficiency-score", "children"),
        Output("utilization-trend-indicator", "children"),
        Output("headcount-trend-indicator", "children"),
        Output("overtime-trend-indicator", "children"),
        Output("efficiency-trend-indicator", "children"),
        Output("utilization-trend-indicator", "className"),
        Output("headcount-trend-indicator", "className"),
        Output("overtime-trend-indicator", "className"),
        Output("efficiency-trend-indicator", "className"),
    ],
    [
        Input("dept-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
    [State("utilization-rate", "children")],
)
def update_kpi_metrics(dept, start_date, end_date, prev_util):
    if not all([start_date, end_date]):
        raise PreventUpdate

    monthly_kpis, weekly_data, _, _, _, _ = load_data()

    # Convert date strings to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Filter data
    filtered_data = weekly_data[
        (weekly_data["date"] >= start_date) & (weekly_data["date"] <= end_date)
    ]

    if dept != "all":
        filtered_data = filtered_data[filtered_data["dept"] == dept]

    # Calculate metrics
    total_direct_hours = filtered_data["direct_hours"].sum()
    total_hours = filtered_data["total_hours_charged"].sum()
    total_overtime = filtered_data["overtime_hours"].sum()
    avg_employee_count = filtered_data["employee_count"].mean()
    unique_dates = len(filtered_data["date"].unique())

    utilization_rate = (
        (total_direct_hours / total_hours * 100) if total_hours > 0 else 0
    )
    overtime_percentage = (total_overtime / total_hours * 100) if total_hours > 0 else 0
    efficiency_score = (
        (total_hours / (avg_employee_count * unique_dates))
        if avg_employee_count > 0 and unique_dates > 0
        else 0
    )

    # Calculate trends
    mid_date = start_date + (end_date - start_date) / 2
    prev_period = filtered_data[filtered_data["date"] < mid_date]
    curr_period = filtered_data[filtered_data["date"] >= mid_date]

    def get_trend_info(curr_val, prev_val):
        if prev_val == 0:
            return "N/A", "trend-neutral"
        elif curr_val > prev_val:
            return "↑ +{:.1f}%".format(((curr_val / prev_val) - 1) * 100), "trend-up"
        else:
            return "↓ {:.1f}%".format(((curr_val / prev_val) - 1) * 100), "trend-down"

    # Calculate current and previous values for trends
    util_curr = (
        (curr_period["direct_hours"].sum() / curr_period["total_hours_charged"].sum())
        if not curr_period.empty
        else 0
    )
    util_prev = (
        (prev_period["direct_hours"].sum() / prev_period["total_hours_charged"].sum())
        if not prev_period.empty
        else 0
    )
    util_trend, util_class = get_trend_info(util_curr, util_prev)

    hc_curr = curr_period["employee_count"].mean() if not curr_period.empty else 0
    hc_prev = prev_period["employee_count"].mean() if not prev_period.empty else 0
    hc_trend, hc_class = get_trend_info(hc_curr, hc_prev)

    ot_curr = (
        (curr_period["overtime_hours"].sum() / curr_period["total_hours_charged"].sum())
        if not curr_period.empty
        else 0
    )
    ot_prev = (
        (prev_period["overtime_hours"].sum() / prev_period["total_hours_charged"].sum())
        if not prev_period.empty
        else 0
    )
    ot_trend, ot_class = get_trend_info(ot_curr, ot_prev)

    eff_curr = (
        (
            curr_period["total_hours_charged"].sum()
            / (curr_period["employee_count"].mean() * len(curr_period["date"].unique()))
        )
        if not curr_period.empty
        else 0
    )
    eff_prev = (
        (
            prev_period["total_hours_charged"].sum()
            / (prev_period["employee_count"].mean() * len(prev_period["date"].unique()))
        )
        if not prev_period.empty
        else 0
    )
    eff_trend, eff_class = get_trend_info(eff_curr, eff_prev)

    return (
        f"{utilization_rate:.1f}%",
        f"{avg_employee_count:.0f}",
        f"{overtime_percentage:.1f}%",
        f"{efficiency_score:.1f}",
        util_trend,
        hc_trend,
        ot_trend,
        eff_trend,
        util_class,
        hc_class,
        ot_class,
        eff_class,
    )


@app.callback(
    [
        Output("utilization-trend", "figure"),
        Output("productivity-metrics", "figure"),
        Output("department-comparison", "figure"),
        Output("efficiency-metrics", "figure"),
        Output("holiday-impact-analysis", "figure"),
    ],
    [
        Input("dept-filter", "value"),
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
    ],
)
def update_overview_charts(dept, start_date, end_date):
    (
        monthly_kpis,
        weekly_data,
        prophet_forecast,
        monte_carlo,
        efficiency_kpis,
        dept_performance,
    ) = load_data()

    # Filter by department if specified
    if dept != "all":
        weekly_data = weekly_data[weekly_data["dept"] == dept]

    # Utilization Trend
    util_fig = px.line(
        weekly_data,
        x="date",
        y="direct_hours",
        color="dept",
        title="Utilization Trend Over Time",
    )
    util_fig.update_layout(
        xaxis_title="Date", yaxis_title="Direct Hours", legend_title="Department"
    )

    # Productivity Metrics
    prod_fig = px.line(
        weekly_data,
        x="date",
        y="total_hours_charged",
        color="dept",
        title="Total Hours Charged Over Time",
    )
    prod_fig.update_layout(
        xaxis_title="Date", yaxis_title="Total Hours", legend_title="Department"
    )

    # Department Comparison
    dept_summary = (
        weekly_data.groupby("dept")
        .agg(
            {
                "total_hours_charged": "sum",
                "direct_hours": "sum",
                "non_direct_hours": "sum",
                "overtime_hours": "sum",
            }
        )
        .reset_index()
    )

    dept_fig = px.bar(
        dept_summary,
        x="dept",
        y=["direct_hours", "non_direct_hours", "overtime_hours"],
        title="Hours Distribution by Department",
        barmode="stack",
    )
    dept_fig.update_layout(
        xaxis_title="Department", yaxis_title="Hours", legend_title="Hour Type"
    )

    # Efficiency Metrics
    efficiency_by_week = (
        weekly_data.groupby(["date", "dept"])
        .agg({"total_hours_charged": "sum", "employee_count": "mean"})
        .reset_index()
    )
    efficiency_by_week["hours_per_employee"] = (
        efficiency_by_week["total_hours_charged"] / efficiency_by_week["employee_count"]
    )

    eff_fig = px.line(
        efficiency_by_week,
        x="date",
        y="hours_per_employee",
        color="dept",
        title="Hours per Employee Over Time",
    )
    eff_fig.update_layout(
        xaxis_title="Date", yaxis_title="Hours per Employee", legend_title="Department"
    )

    # Instead of holiday impact, let's show overtime analysis
    overtime_fig = px.line(
        weekly_data.groupby(["date", "dept"])["overtime_hours"].sum().reset_index(),
        x="date",
        y="overtime_hours",
        color="dept",
        title="Overtime Hours Trend",
    )
    overtime_fig.update_layout(
        xaxis_title="Date", yaxis_title="Overtime Hours", legend_title="Department"
    )

    return util_fig, prod_fig, dept_fig, eff_fig, overtime_fig


@app.callback(
    [
        Output("dept-metrics-chart", "figure"),
        Output("dept-comparison-chart", "figure"),
        Output("dept-performance-trend", "figure"),
    ],
    [
        Input("dept-metrics-selector", "value"),
        Input("tabs", "value"),
    ],
)
def update_department_charts(selected_metric, tab):
    if tab != "department":
        raise PreventUpdate

    monthly_kpis, weekly_data, _, _, efficiency_kpis, dept_performance = load_data()

    # Department Metrics Chart
    metrics_map = {
        "utilization": "direct_hours",
        "overtime": "overtime_hours",
        "direct_labor": "total_hours_charged",
    }

    metric_col = metrics_map[selected_metric]
    dept_metrics_fig = px.line(
        weekly_data,
        x="date",
        y=metric_col,
        color="dept",
        title=f"Department {selected_metric.title()} Over Time",
    )

    # Department Comparison Chart
    # First, let's print the structure of dept_performance to debug
    print("Department Performance columns:", dept_performance.columns)

    # Create a simpler comparison using weekly_data instead
    dept_summary = (
        weekly_data.groupby("dept")
        .agg(
            {
                "total_hours_charged": "sum",
                "direct_hours": "sum",
                "overtime_hours": "sum",
            }
        )
        .reset_index()
    )

    dept_comparison_fig = px.bar(
        dept_summary,
        x="dept",
        y=["total_hours_charged", "direct_hours", "overtime_hours"],
        title="Department Performance Comparison",
        barmode="group",
    )

    # Department Performance Trend
    performance_trend_fig = px.line(
        weekly_data.groupby(["date", "dept"])["total_hours_charged"]
        .mean()
        .reset_index(),
        x="date",
        y="total_hours_charged",
        color="dept",
        title="Department Performance Trends",
    )

    return dept_metrics_fig, dept_comparison_fig, performance_trend_fig


# Forecasting Tab Layout
def create_forecasting_layout():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Labor Hours Forecast"),
                            dcc.Graph(id="labor-forecast-chart"),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Monte Carlo Simulation"),
                            dcc.Graph(id="monte-carlo-chart"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Confidence Intervals"),
                            dcc.Graph(id="forecast-confidence-chart"),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


# Employee Metrics Tab Layout
def create_employee_layout():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Department Performance Metrics (Per Employee)"),
                            dcc.Dropdown(
                                id="employee-metric-selector",
                                options=[
                                    {
                                        "label": "Total Hours per Employee",
                                        "value": "total_hours_charged",
                                    },
                                    {
                                        "label": "Direct Hours per Employee",
                                        "value": "direct_hours",
                                    },
                                    {
                                        "label": "Non-Direct Hours per Employee",
                                        "value": "non_direct_hours",
                                    },
                                ],
                                value="total_hours_charged",
                            ),
                            dcc.Graph(id="employee-performance-chart"),
                        ],
                        width=12,
                    ),
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H4("Department Utilization Distribution"),
                            dcc.Graph(id="employee-utilization-chart"),
                        ],
                        width=6,
                    ),
                    dbc.Col(
                        [
                            html.H4("Average Overtime per Employee"),
                            dcc.Graph(id="employee-overtime-chart"),
                        ],
                        width=6,
                    ),
                ]
            ),
        ]
    )


# Forecasting Tab Callbacks
@app.callback(
    [
        Output("labor-forecast-chart", "figure"),
        Output("monte-carlo-chart", "figure"),
        Output("forecast-confidence-chart", "figure"),
    ],
    [Input("tabs", "value")],
)
def update_forecasting_charts(tab):
    if tab != "forecasting":
        raise PreventUpdate

    _, _, prophet_forecast, monte_carlo, _, _ = load_data()

    # Print data structure to debug
    print("Prophet forecast columns:", prophet_forecast.columns)
    print("Monte Carlo columns:", monte_carlo.columns)

    # Labor Hours Forecast
    forecast_fig = px.line(
        prophet_forecast,
        x="date",  # Changed from 'ds' to 'date'
        y=["actual_hours", "forecast"],  # Changed from 'y' and 'yhat'
        title="Labor Hours Forecast",
        labels={"date": "Date", "actual_hours": "Actual Hours", "forecast": "Forecast"},
    )

    # Monte Carlo Simulation
    monte_carlo_fig = px.line(
        monte_carlo,
        x="date",
        y=["mc_mean", "mc_lower", "mc_upper"],
        title="Monte Carlo Simulation Results",
        labels={
            "date": "Date",
            "mc_mean": "Mean Forecast",
            "mc_lower": "Lower Bound",
            "mc_upper": "Upper Bound",
        },
    )

    # Confidence Intervals
    confidence_fig = go.Figure()
    confidence_fig.add_trace(
        go.Scatter(
            x=prophet_forecast["date"],  # Changed from 'ds'
            y=prophet_forecast["forecast_upper"],  # Changed from 'yhat_upper'
            fill=None,
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="Upper Bound",
        )
    )
    confidence_fig.add_trace(
        go.Scatter(
            x=prophet_forecast["date"],  # Changed from 'ds'
            y=prophet_forecast["forecast_lower"],  # Changed from 'yhat_lower'
            fill="tonexty",
            mode="lines",
            line_color="rgba(0,100,80,0.2)",
            name="Lower Bound",
        )
    )
    confidence_fig.add_trace(
        go.Scatter(
            x=prophet_forecast["date"],  # Changed from 'ds'
            y=prophet_forecast["forecast"],  # Changed from 'yhat'
            mode="lines",
            line_color="rgb(0,100,80)",
            name="Forecast",
        )
    )
    confidence_fig.update_layout(title="Forecast Confidence Intervals")

    return forecast_fig, monte_carlo_fig, confidence_fig


# Employee Metrics Tab Callbacks
@app.callback(
    [
        Output("employee-performance-chart", "figure"),
        Output("employee-utilization-chart", "figure"),
        Output("employee-overtime-chart", "figure"),
    ],
    [Input("employee-metric-selector", "value"), Input("tabs", "value")],
)
def update_employee_charts(selected_metric, tab):
    if tab != "employee":
        raise PreventUpdate

    _, weekly_data, _, _, _, _ = load_data()

    # Calculate per-employee metrics first
    plot_data = weekly_data.copy()
    plot_data[f"{selected_metric}_per_employee"] = (
        plot_data[selected_metric] / plot_data["employee_count"]
    )

    # Employee Performance Chart (by department)
    performance_fig = px.line(
        plot_data,
        x="date",
        y=f"{selected_metric}_per_employee",
        color="dept",
        title=f"Department {selected_metric.replace('_', ' ').title()} Per Employee",
    )

    # Utilization Distribution by Department
    plot_data["utilization_rate"] = (
        plot_data["direct_hours"] / plot_data["total_hours_charged"] * 100
    )

    utilization_fig = px.box(
        plot_data,
        x="dept",
        y="utilization_rate",
        title="Department Utilization Rate Distribution",
        labels={"utilization_rate": "Utilization Rate (%)", "dept": "Department"},
    )

    # Overtime Trends by Department (per employee)
    plot_data["overtime_per_employee"] = (
        plot_data["overtime_hours"] / plot_data["employee_count"]
    )

    overtime_fig = px.line(
        plot_data,
        x="date",
        y="overtime_per_employee",
        color="dept",
        title="Average Overtime Hours per Employee by Department",
        labels={
            "overtime_per_employee": "Overtime Hours per Employee",
            "dept": "Department",
        },
    )

    # Update layout for all figures
    for fig in [performance_fig, utilization_fig, overtime_fig]:
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Date",
            legend_title="Department",
            height=400,
        )

    return performance_fig, utilization_fig, overtime_fig


# Add these styles to your CSS
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Labor KPI Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            .kpi-card {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
                transition: transform 0.2s;
            }
            .kpi-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            .kpi-value {
                font-size: 2em;
                font-weight: bold;
                color: #2c3e50;
                margin: 10px 0;
            }
            .trend-indicator {
                font-size: 0.9em;
                margin: 5px 0;
            }
            .trend-up {
                color: #27ae60;
                font-weight: bold;
            }
            .trend-down {
                color: #c0392b;
                font-weight: bold;
            }
            .card-title {
                color: #7f8c8d;
                font-size: 1.1em;
                margin-bottom: 0;
            }
            .tooltip {
                font-size: 0.9em;
                max-width: 200px;
            }
            .kpi-container {
                margin: 20px 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

if __name__ == "__main__":
    app.run_server(debug=True)
