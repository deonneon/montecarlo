import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Initialize the Dash app
app = dash.Dash(
    __name__, external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"]
)

# Load the data
df = pd.read_csv("labor_data.csv")
df["date"] = pd.to_datetime(df["date"])
prophet_forecast = pd.read_csv("prophet_forecast_data.csv")
monte_carlo = pd.read_csv("monte_carlo_results.csv")

# Layout
app.layout = html.Div(
    [
        # Header
        html.H1("Labor Analytics Dashboard", style={"textAlign": "center"}),
        # Filters Row
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Date Range"),
                        dcc.DatePickerRange(
                            id="date-range",
                            start_date=df["date"].min(),
                            end_date=df["date"].max(),
                            display_format="YYYY-MM-DD",
                        ),
                    ],
                    className="three columns",
                ),
                html.Div(
                    [
                        html.Label("Department"),
                        dcc.Dropdown(
                            id="dept-filter",
                            options=[
                                {"label": dept, "value": dept}
                                for dept in df["dept"].unique()
                            ],
                            multi=True,
                        ),
                    ],
                    className="three columns",
                ),
                html.Div(
                    [
                        html.Label("Metric Type"),
                        dcc.Dropdown(
                            id="metric-type",
                            options=[
                                {
                                    "label": "Total Hours",
                                    "value": "total_hours_charged",
                                },
                                {"label": "Direct Hours", "value": "direct_hours"},
                                {"label": "Overtime Hours", "value": "overtime_hours"},
                            ],
                            value="total_hours_charged",
                        ),
                    ],
                    className="three columns",
                ),
            ],
            className="row",
        ),
        # Main Charts Row
        html.Div(
            [
                # Labor Hours Overview
                html.Div(
                    [html.H3("Labor Hours Overview"), dcc.Graph(id="main-time-series")],
                    className="eight columns",
                ),
                # KPIs
                html.Div(
                    [html.H3("Current Period KPIs"), html.Div(id="kpi-metrics")],
                    className="four columns",
                ),
            ],
            className="row",
        ),
        # Second Row of Charts
        html.Div(
            [
                # Department Performance
                html.Div(
                    [
                        html.H3("Department Performance"),
                        dcc.Graph(id="dept-performance"),
                    ],
                    className="six columns",
                ),
                # Employee Distribution
                html.Div(
                    [
                        html.H3("Employee Distribution"),
                        dcc.Graph(id="employee-heatmap"),
                    ],
                    className="six columns",
                ),
            ],
            className="row",
        ),
        # Forecast Row
        html.Div(
            [html.H3("30-Day Forecast"), dcc.Graph(id="forecast-chart")],
            className="row",
        ),
    ]
)


# Callback for main time series
@app.callback(
    Output("main-time-series", "figure"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("dept-filter", "value"),
        Input("metric-type", "value"),
    ],
)
def update_time_series(start_date, end_date, departments, metric):
    # Filter data
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    if departments:
        mask = mask & df["dept"].isin(departments)
    filtered_df = df[mask]

    # Create figure
    fig = go.Figure()

    # Add actual data
    fig.add_trace(
        go.Scatter(
            x=filtered_df["date"],
            y=filtered_df.groupby("date")[metric].sum(),
            name="Actual",
            mode="lines",
        )
    )

    # Add forecast if available
    if metric == "total_hours_charged":
        fig.add_trace(
            go.Scatter(
                x=prophet_forecast["date"],
                y=prophet_forecast["forecast"],
                name="Forecast",
                mode="lines",
                line=dict(dash="dash"),
            )
        )

        # Add confidence intervals
        fig.add_trace(
            go.Scatter(
                x=prophet_forecast["date"],
                y=prophet_forecast["forecast_upper"],
                fill=None,
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=prophet_forecast["date"],
                y=prophet_forecast["forecast_lower"],
                fill="tonexty",
                mode="lines",
                line_color="rgba(0,100,80,0.2)",
                name="95% Confidence Interval",
            )
        )

    fig.update_layout(
        title=f'{metric.replace("_", " ").title()} Over Time',
        xaxis_title="Date",
        yaxis_title="Hours",
        hovermode="x unified",
    )

    return fig


# Callback for KPI metrics
@app.callback(
    Output("kpi-metrics", "children"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("dept-filter", "value"),
    ],
)
def update_kpis(start_date, end_date, departments):
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    if departments:
        mask = mask & df["dept"].isin(departments)
    filtered_df = df[mask]

    utilization_rate = (
        filtered_df["direct_hours"].sum()
        / filtered_df["total_hours_charged"].sum()
        * 100
    )
    overtime_rate = (
        filtered_df["overtime_hours"].sum()
        / filtered_df["total_hours_charged"].sum()
        * 100
    )

    return html.Div(
        [
            html.P(f"Utilization Rate: {utilization_rate:.1f}%"),
            html.P(f"Overtime Rate: {overtime_rate:.1f}%"),
            html.P(f'Active Employees: {filtered_df["userid"].nunique()}'),
        ]
    )


# Callback for department performance
@app.callback(
    Output("dept-performance", "figure"),
    [
        Input("date-range", "start_date"),
        Input("date-range", "end_date"),
        Input("metric-type", "value"),
    ],
)
def update_dept_performance(start_date, end_date, metric):
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered_df = df[mask]

    dept_perf = filtered_df.groupby("dept")[metric].sum().reset_index()

    fig = px.bar(
        dept_perf,
        x="dept",
        y=metric,
        title=f'Department Performance - {metric.replace("_", " ").title()}',
    )

    return fig


# Callback for employee heatmap
@app.callback(
    Output("employee-heatmap", "figure"),
    [Input("date-range", "start_date"), Input("date-range", "end_date")],
)
def update_employee_heatmap(start_date, end_date):
    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    filtered_df = df[mask]

    # Create monthly aggregation
    filtered_df["month"] = filtered_df["date"].dt.to_period("M")
    heatmap_data = filtered_df.pivot_table(
        values="total_hours_charged", index="dept", columns="month", aggfunc="sum"
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.astype(str),
            y=heatmap_data.index,
            colorscale="Viridis",
        )
    )

    fig.update_layout(
        title="Employee Hours Distribution",
        xaxis_title="Month",
        yaxis_title="Department",
    )

    return fig


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
