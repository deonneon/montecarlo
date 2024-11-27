import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import clientside_callback, ClientsideFunction
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data_processing import DataPreprocessor
from time_series_model import TimeSeriesPredictor
from monte_carlo import MonteCarloSimulator
from worker_monte_carlo import WorkerMonteCarloPredictor
from plotly.subplots import make_subplots
from hybrid_predictor import HybridLaborPredictor

# Initialize the Dash app
app = dash.Dash(
    __name__, title="Labor Analytics Dashboard", suppress_callback_exceptions=True
)

# Initialize components
preprocessor = DataPreprocessor(fiscal_start_month=10)
predictor = TimeSeriesPredictor()
simulator = MonteCarloSimulator()

# Load and prepare data
raw_data = preprocessor.load_and_prepare_data("data/generated/labor_data.csv")
daily_data, dept_daily_data = preprocessor.aggregate_daily_metrics(raw_data)
daily_data = preprocessor.create_manufacturing_features(daily_data)

# Dashboard layout
app.layout = html.Div(
    [
        # Header
        html.H1(
            "Manufacturing Labor Analytics Dashboard",
            style={
                "textAlign": "center",
                "marginBottom": "30px",
                "padding": "20px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "10px",
            },
        ),
        # Filters Container
        html.Div(
            [
                # Date Range Filter
                html.Div(
                    [
                        html.Label(
                            "Date Range:",
                            style={
                                "marginBottom": "10px",
                                "fontWeight": "bold",
                                "display": "block",
                            },
                        ),
                        dcc.DatePickerRange(
                            id="date-picker",
                            start_date=daily_data["date"].min(),
                            end_date=daily_data["date"].max(),
                            display_format="YYYY-MM-DD",
                            style={"width": "100%"},
                        ),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "marginRight": "20px",
                        "verticalAlign": "top",
                    },
                ),
                # Department Filter
                html.Div(
                    [
                        html.Label(
                            "Department:",
                            style={
                                "marginBottom": "10px",
                                "fontWeight": "bold",
                                "display": "block",
                            },
                        ),
                        dcc.Dropdown(
                            id="department-filter",
                            options=[{"label": "All Departments", "value": "All"}]
                            + [
                                {"label": dept, "value": dept}
                                for dept in dept_daily_data["dept"].unique()
                            ],
                            value="All",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
            ],
            style={
                "margin": "20px",
                "padding": "20px",
                "backgroundColor": "#f8f9fa",
                "borderRadius": "10px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
        ),
        # KPI Cards Container
        html.Div(
            [
                # Total Hours Card
                html.Div(
                    [
                        html.H4("Total Hours", style={"margin": "0", "color": "#666"}),
                        html.H2(
                            id="total-hours",
                            style={"margin": "10px 0 0 0", "color": "#333"},
                        ),
                    ],
                    className="kpi-card",
                ),
                # Labor Efficiency Card
                html.Div(
                    [
                        html.H4(
                            "Labor Efficiency", style={"margin": "0", "color": "#666"}
                        ),
                        html.H2(
                            id="efficiency",
                            style={"margin": "10px 0 0 0", "color": "#333"},
                        ),
                    ],
                    className="kpi-card",
                ),
                # Overtime Card
                html.Div(
                    [
                        html.H4(
                            "Overtime Ratio", style={"margin": "0", "color": "#666"}
                        ),
                        html.H2(
                            id="overtime",
                            style={"margin": "10px 0 0 0", "color": "#333"},
                        ),
                    ],
                    className="kpi-card",
                ),
            ],
            style={
                "display": "flex",
                "justifyContent": "space-around",
                "margin": "20px",
                "padding": "10px",
            },
        ),
        # Charts Container
        html.Div(
            [
                # Labor Trend Chart
                html.Div(
                    dcc.Graph(id="labor-trend"),
                    style={
                        "width": "100%",
                        "marginBottom": "20px",
                        "backgroundColor": "#fff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "padding": "15px",
                    },
                ),
                # Bottom Charts Container
                html.Div(
                    [
                        # Department Comparison Chart
                        html.Div(
                            dcc.Graph(id="dept-comparison"),
                            style={
                                "width": "48%",
                                "display": "inline-block",
                                "backgroundColor": "#fff",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "padding": "15px",
                                "marginRight": "2%",
                            },
                        ),
                        # Forecast Chart
                        html.Div(
                            dcc.Graph(id="forecast"),
                            style={
                                "width": "48%",
                                "display": "inline-block",
                                "backgroundColor": "#fff",
                                "borderRadius": "10px",
                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "padding": "15px",
                            },
                        ),
                    ],
                    style={"display": "flex", "justifyContent": "space-between"},
                ),
                # Fiscal Year Results
                html.Div(
                    dcc.Graph(id="fiscal-year-plot"),
                    style={
                        "width": "100%",
                        "marginTop": "20px",
                        "marginBottom": "20px",
                        "backgroundColor": "#fff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "padding": "15px",
                    },
                ),
                # Fiscal Pattern Plot
                html.Div(
                    dcc.Graph(id="fiscal-pattern-plot"),
                    style={
                        "backgroundColor": "#fff",
                        "borderRadius": "10px",
                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                        "padding": "15px",
                    },
                ),
            ],
            style={"margin": "20px", "padding": "10px"},
        ),
        html.Div(
            [
                html.H3("Individual Worker Predictions"),
                dcc.Dropdown(
                    id="worker-selector",
                    options=[
                        {"label": worker, "value": worker}
                        for worker in raw_data["userid"].unique()
                    ],
                    value=raw_data["userid"].iloc[0],
                ),
                dcc.Graph(id="worker-predictions"),
            ],
            style={"margin": "20px", "padding": "20px"},
        ),
        # Add this to the dashboard layout after the individual worker predictions
        html.Div(
            [
                html.H3("Aggregate Workforce Predictions"),
                dcc.Graph(id="aggregated-worker-predictions"),
            ],
            style={"margin": "20px", "padding": "20px"},
        ),
        # Add this after the other charts
        html.Div(
            [
                html.H3("Average Hours per Worker"),
                dcc.Graph(id="average-hours-predictions"),
            ],
            style={"margin": "20px", "padding": "20px"},
        ),
        # Hybrid Forecast Section
        html.Div(
            [
                # Header for Hybrid Forecast Section
                html.H3(
                    "Hybrid Labor Forecast Analysis",
                    style={"marginBottom": "20px", "textAlign": "center"},
                ),
                # Control Panel
                html.Div(
                    [
                        # Forecast Settings
                        html.Div(
                            [
                                html.Label(
                                    "Forecast Settings:",
                                    style={
                                        "fontWeight": "bold",
                                        "marginBottom": "10px",
                                    },
                                ),
                                dcc.Checklist(
                                    id="forecast-options",
                                    options=[
                                        {
                                            "label": " Include Seasonality",
                                            "value": "seasonality",
                                        },
                                        {
                                            "label": " Include Growth Projections",
                                            "value": "growth",
                                        },
                                    ],
                                    value=["seasonality", "growth"],
                                    style={"marginBottom": "15px"},
                                ),
                            ],
                            style={
                                "width": "30%",
                                "display": "inline-block",
                                "marginRight": "20px",
                            },
                        ),
                        # Forecast Horizon Selector
                        html.Div(
                            [
                                html.Label(
                                    "Forecast Horizon:",
                                    style={
                                        "fontWeight": "bold",
                                        "marginBottom": "10px",
                                    },
                                ),
                                dcc.Slider(
                                    id="forecast-horizon",
                                    min=7,
                                    max=90,
                                    step=7,
                                    value=30,
                                    marks={i: f"{i}d" for i in range(7, 91, 7)},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": True,
                                    },
                                ),
                            ],
                            style={"width": "60%", "display": "inline-block"},
                        ),
                    ],
                    style={
                        "marginBottom": "20px",
                        "padding": "15px",
                        "backgroundColor": "#f8f9fa",
                        "borderRadius": "10px",
                    },
                ),
                # Main Forecast Chart
                html.Div(
                    [dcc.Graph(id="hybrid-forecast-chart")],
                    style={"marginBottom": "20px"},
                ),
                # Metrics Row
                html.Div(
                    [
                        # Confidence Metrics Card
                        html.Div(
                            [
                                html.H5("Confidence Metrics"),
                                html.Div(
                                    id="confidence-metrics", className="metric-content"
                                ),
                            ],
                            className="metric-card",
                        ),
                        # Model Weights Card
                        html.Div(
                            [
                                html.H5("Model Weights"),
                                html.Div(
                                    id="model-weights", className="metric-content"
                                ),
                            ],
                            className="metric-card",
                        ),
                        # Forecast Accuracy Card
                        html.Div(
                            [
                                html.H5("Forecast Accuracy"),
                                html.Div(
                                    id="forecast-accuracy", className="metric-content"
                                ),
                            ],
                            className="metric-card",
                        ),
                    ],
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "marginBottom": "20px",
                    },
                ),
                # Department Breakdown
                html.Div(
                    [
                        html.H4(
                            "Department-Level Forecasts",
                            style={"marginBottom": "15px"},
                        ),
                        dcc.Graph(id="dept-forecast-breakdown"),
                    ],
                    style={"marginBottom": "20px"},
                ),
            ],
            style={
                "margin": "20px",
                "padding": "20px",
                "backgroundColor": "#ffffff",
                "borderRadius": "10px",
                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
            },
        ),
    ],
    style={"backgroundColor": "#f0f2f5", "minHeight": "100vh", "padding": "20px"},
)

# Add this after your app layout
app.clientside_callback(
    ClientsideFunction(namespace="clientside", function_name="toggle_growth"),
    Output("hybrid-forecast-chart", "figure", allow_duplicate=True),
    Input("forecast-options", "value"),
    State("hybrid-forecast-chart", "figure"),
    prevent_initial_call=True,
)


# Update KPIs callback
@app.callback(
    [
        Output("total-hours", "children"),
        Output("efficiency", "children"),
        Output("overtime", "children"),
    ],
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_kpis(start_date, end_date, selected_dept):
    filtered_data, _ = filter_data(start_date, end_date, selected_dept)

    total_hours = f"{filtered_data['total_hours_charged'].sum():,.0f}"

    direct_hours = filtered_data["direct_hours"].sum()
    total_charged = filtered_data["total_hours_charged"].sum()
    efficiency = (
        f"{(direct_hours / total_charged * 100):.1f}%" if total_charged > 0 else "0.0%"
    )

    overtime = (
        f"{(filtered_data['overtime_hours'].sum() / total_charged * 100):.1f}%"
        if total_charged > 0
        else "0.0%"
    )

    return total_hours, efficiency, overtime


# Update labor trend callback
@app.callback(
    Output("labor-trend", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_labor_trend(start_date, end_date, selected_dept):
    filtered_data, _ = filter_data(start_date, end_date, selected_dept)
    print("Available columns:", filtered_data.columns)  # Debug print

    # Create subplot figure with two rows
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Total Labor Hours", "Average Hours per Worker"),
        vertical_spacing=0.12,
    )

    # Plot 1: Total Hours
    fig.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["total_hours_charged"],
            mode="lines",
            name="Total Hours",
            line=dict(color="#2ecc71"),
        ),
        row=1,
        col=1,
    )

    # Plot 2: Average Hours per Worker
    # Calculate daily averages using employee_count instead of userid
    daily_data = (
        filtered_data.groupby("date")
        .agg(
            {
                "total_hours_charged": "sum",
                "employee_count": "first",  # Changed from userid to employee_count
            }
        )
        .reset_index()
    )

    daily_data["avg_hours_per_worker"] = (
        daily_data["total_hours_charged"] / daily_data["employee_count"]
    )

    fig.add_trace(
        go.Scatter(
            x=daily_data["date"],
            y=daily_data["avg_hours_per_worker"],
            mode="lines",
            name="Avg Hours per Worker",
            line=dict(color="#3498db"),
        ),
        row=2,
        col=1,
    )

    # Add 8-hour reference line to average hours plot
    fig.add_hline(
        y=8,
        line_dash="dash",
        line_color="red",
        annotation_text="8 Hour Standard",
        annotation_position="right",
        row=2,
    )

    # Update layout
    title = (
        "Labor Hours Trend - All Departments"
        if selected_dept == "All"
        else f"Labor Hours Trend - {selected_dept}"
    )

    fig.update_layout(
        height=800,  # Increased height for two plots
        title=title,
        showlegend=True,
        template="plotly_white",
        hovermode="x unified",
    )

    # Update x-axis labels
    fig.update_xaxes(title_text="Date", row=2, col=1)

    # Update y-axis labels
    fig.update_yaxes(title_text="Total Hours", row=1, col=1)
    fig.update_yaxes(title_text="Average Hours per Worker", row=2, col=1)

    return fig


# Department comparison callback
@app.callback(
    Output("dept-comparison", "figure"),
    [Input("date-picker", "start_date"), Input("date-picker", "end_date")],
)
def update_dept_comparison(start_date, end_date):
    # Filter data by date range
    mask = (dept_daily_data["date"] >= start_date) & (
        dept_daily_data["date"] <= end_date
    )
    filtered_data = dept_daily_data[mask].copy()

    # Calculate average daily hours by department
    dept_avg = (
        filtered_data.groupby("dept")
        .agg(
            {
                "total_hours_charged": "mean",
                "direct_hours": "mean",
                "overtime_hours": "mean",
            }
        )
        .reset_index()
    )

    # Create stacked bar chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Direct Hours",
            x=dept_avg["dept"],
            y=dept_avg["direct_hours"],
            marker_color="#2ecc71",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Non-Direct Hours",
            x=dept_avg["dept"],
            y=dept_avg["total_hours_charged"] - dept_avg["direct_hours"],
            marker_color="#3498db",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Overtime Hours",
            x=dept_avg["dept"],
            y=dept_avg["overtime_hours"],
            marker_color="#e74c3c",
        )
    )

    fig.update_layout(
        title="Average Daily Hours by Department",
        barmode="stack",
        xaxis_title="Department",
        yaxis_title="Hours",
        height=400,
        template="plotly_white",
    )

    return fig


# Update forecast callback
@app.callback(
    Output("forecast", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_forecast(start_date, end_date, selected_dept):
    filtered_data, _ = filter_data(start_date, end_date, selected_dept)

    # Fit time series model
    ts_data = filtered_data["total_hours_charged"].values
    predictor.fit_holtwinters(ts_data)

    # Generate forecast for next 30 days
    forecast_horizon = 30
    mean_forecast, lower_bound, upper_bound = simulator.generate_scenarios(
        predictor.model, filtered_data, forecast_horizon
    )

    # Create forecast dates
    last_date = pd.to_datetime(filtered_data["date"].max())
    forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["total_hours_charged"],
            name="Historical",
            line=dict(color="#2ecc71"),
        )
    )

    # Add forecast
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=mean_forecast,
            name="Forecast",
            line=dict(color="#e74c3c", dash="dash"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper Bound",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(231, 76, 60, 0.2)",
            fill="tonexty",
            showlegend=False,
            name="Lower Bound",
        )
    )

    title = (
        "30-Day Labor Hours Forecast"
        if selected_dept == "All"
        else f"30-Day Labor Hours Forecast - {selected_dept}"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Hours",
        height=400,
        template="plotly_white",
    )

    return fig


# Update fiscal year plot callback
@app.callback(
    Output("fiscal-year-plot", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_fiscal_year_plot(start_date, end_date, selected_dept):
    filtered_data, _ = filter_data(start_date, end_date, selected_dept)

    # Fit time series model
    ts_data = filtered_data["total_hours_charged"].values
    predictor.fit_holtwinters(ts_data)

    # Generate forecast for next fiscal year (approximately 252 trading days)
    forecast_horizon = 252
    mean_forecast, lower_bound, upper_bound = simulator.generate_scenarios(
        predictor.model, filtered_data, forecast_horizon
    )

    # Create forecast dates
    last_date = pd.to_datetime(filtered_data["date"].max())
    forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[1:]

    fig = go.Figure()

    # Plot historical data
    fig.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["total_hours_charged"],
            name="Historical Data",
            line=dict(color="blue"),
        )
    )

    # Add forecast and confidence intervals
    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=mean_forecast,
            name="Forecast",
            line=dict(color="#e74c3c", dash="dash"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=upper_bound,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_dates,
            y=lower_bound,
            mode="lines",
            line=dict(width=0),
            fillcolor="rgba(231, 76, 60, 0.2)",
            fill="tonexty",
            showlegend=False,
        )
    )

    # Add fiscal year boundaries
    fiscal_years = filtered_data["fiscal_year"].unique()
    for fy in fiscal_years:
        fy_start = pd.Timestamp(f"{fy}-10-01")  # October 1st
        if (
            fy_start >= filtered_data["date"].min()
            and fy_start <= filtered_data["date"].max()
        ):
            fig.add_trace(
                go.Scatter(
                    x=[fy_start, fy_start],
                    y=[
                        filtered_data["total_hours_charged"].min(),
                        filtered_data["total_hours_charged"].max(),
                    ],
                    mode="lines",
                    line=dict(color="gray", dash="dash"),
                    name=f"FY{fy} Start",
                    showlegend=False,
                )
            )

            fig.add_annotation(
                x=fy_start,
                y=filtered_data["total_hours_charged"].max(),
                text=f"FY{fy}",
                showarrow=False,
                yshift=10,
            )

    fig.update_layout(
        title="Labor Hours by Fiscal Year with Forecast",
        xaxis_title="Date",
        yaxis_title="Total Hours Charged",
        height=400,
        template="plotly_white",
    )

    return fig


# Update fiscal pattern plot callback
@app.callback(
    Output("fiscal-pattern-plot", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_fiscal_pattern_plot(start_date, end_date, selected_dept):
    filtered_data, _ = filter_data(start_date, end_date, selected_dept)

    # Calculate mean and std by fiscal period
    fiscal_pattern = (
        filtered_data.groupby("fiscal_period")["total_hours_charged"]
        .agg(["mean", "std"])
        .reset_index()
    )

    fig = go.Figure()

    # Add bar chart with error bars
    fig.add_trace(
        go.Bar(
            x=fiscal_pattern["fiscal_period"],
            y=fiscal_pattern["mean"],
            error_y=dict(type="data", array=fiscal_pattern["std"], visible=True),
            name="Average Hours",
        )
    )

    fig.update_layout(
        title="Average Labor Hours by Fiscal Period",
        xaxis_title="Fiscal Period (1 = October)",
        yaxis_title="Average Hours",
        height=400,
        template="plotly_white",
        showlegend=True,
    )

    # Update x-axis to show all periods
    fig.update_xaxes(tickmode="linear", tick0=1, dtick=1)

    return fig


@app.callback(
    Output("worker-predictions", "figure"),
    [Input("worker-selector", "value"), Input("department-filter", "value")],
)
def update_worker_predictions(selected_worker, selected_dept):
    # Initialize predictor
    worker_predictor = WorkerMonteCarloPredictor()

    # Get historical data for selected worker
    worker_data = raw_data[raw_data["userid"] == selected_worker].copy()
    worker_data["date"] = pd.to_datetime(worker_data["date"])

    # Get last 60 days of data
    last_date = worker_data["date"].max()
    start_date = last_date - pd.Timedelta(days=60)
    historical_data = worker_data[worker_data["date"] >= start_date]

    # Get predictions for selected worker
    simulations, summary = worker_predictor.predict_worker_next_week(
        raw_data, selected_worker
    )

    # Create dates for next 5 work days
    next_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=5, freq="B"  # Business days
    )

    # Create figure
    fig = go.Figure()

    # Add historical data
    fig.add_trace(
        go.Scatter(
            x=historical_data["date"],
            y=historical_data["total_hours_charged"],
            name="Historical Hours",
            line=dict(color="#2ecc71"),
        )
    )

    # Add mean prediction line
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=summary["mean_prediction"],
            name="Mean Prediction",
            line=dict(color="#3498db", dash="dash"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=summary["upper_bound"],
            mode="lines",
            name="95% CI Upper",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=summary["lower_bound"],
            mode="lines",
            name="95% CI Lower",
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Add markers for overtime days (>8 hours) in historical data
    overtime_data = historical_data[historical_data["total_hours_charged"] > 8.0]
    if not overtime_data.empty:
        fig.add_trace(
            go.Scatter(
                x=overtime_data["date"],
                y=overtime_data["total_hours_charged"],
                mode="markers",
                name="Overtime Days (>8hrs)",
                marker=dict(color="#e74c3c", size=8, symbol="star"),
            )
        )

    # Calculate worker statistics for the title
    avg_hours = historical_data["total_hours_charged"].mean()
    overtime_days = len(overtime_data)
    dept = summary["worker_stats"]["department"]

    # Add a shape for the vertical line
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        ),
    )

    # Add annotation for the vertical line
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Prediction Start",
        showarrow=False,
        textangle=-90,
        xanchor="left",
        yanchor="bottom",
    )

    fig.update_layout(
        title=f"{selected_worker} - {dept}<br>"
        f"Avg Hours: {avg_hours:.1f} | Days Over 8hrs: {overtime_days}",
        xaxis_title="Date",
        yaxis_title="Hours",
        height=500,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Ensure y-axis starts at 0 and has some padding at the top
    y_max = max(
        historical_data["total_hours_charged"].max(), summary["upper_bound"].max()
    )
    fig.update_yaxes(range=[0, y_max * 1.1])

    return fig


@app.callback(
    Output("aggregated-worker-predictions", "figure"),
    [Input("department-filter", "value")],
)
def update_aggregated_predictions(selected_dept):
    # Initialize predictor
    worker_predictor = WorkerMonteCarloPredictor()

    # Get all worker predictions
    worker_predictions = worker_predictor.predict_all_workers(raw_data)

    # Get historical data aggregated by date
    historical_data = raw_data.copy()
    historical_data["date"] = pd.to_datetime(historical_data["date"])

    # Get last 60 days of data
    last_date = historical_data["date"].max()
    start_date = last_date - pd.Timedelta(days=60)
    historical_data = historical_data[historical_data["date"] >= start_date]

    # Aggregate historical data by date
    daily_totals = (
        historical_data.groupby("date")
        .agg({"total_hours_charged": "sum", "userid": "nunique"})
        .reset_index()
    )

    # Aggregate predictions for next 5 days
    next_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq="B")

    # Sum up predictions across all workers
    total_mean_prediction = np.zeros(5)
    total_lower_bound = np.zeros(5)
    total_upper_bound = np.zeros(5)

    for worker_id, prediction in worker_predictions.items():
        total_mean_prediction += prediction["mean_prediction"]
        total_lower_bound += prediction["lower_bound"]
        total_upper_bound += prediction["upper_bound"]

    # Create figure
    fig = go.Figure()

    # Add historical total hours
    fig.add_trace(
        go.Scatter(
            x=daily_totals["date"],
            y=daily_totals["total_hours_charged"],
            name="Historical Total Hours",
            line=dict(color="#2ecc71"),
        )
    )

    # Add predicted total hours
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=total_mean_prediction,
            name="Predicted Total Hours",
            line=dict(color="#3498db", dash="dash"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=total_upper_bound,
            mode="lines",
            name="95% CI Upper",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=total_lower_bound,
            mode="lines",
            name="95% CI Lower",
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Add a shape for the vertical line
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        ),
    )

    # Add annotation for the vertical line
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Prediction Start",
        showarrow=False,
        textangle=-90,
        xanchor="left",
        yanchor="bottom",
    )

    # Calculate summary statistics
    avg_total_hours = daily_totals["total_hours_charged"].mean()
    avg_workers = daily_totals["userid"].mean()

    fig.update_layout(
        title=f"Aggregate Workforce Hours<br>"
        f"Avg Daily Hours: {avg_total_hours:.1f} | Avg Workers: {avg_workers:.1f}",
        xaxis_title="Date",
        yaxis_title="Total Hours",
        height=500,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


@app.callback(
    Output("average-hours-predictions", "figure"), [Input("department-filter", "value")]
)
def update_average_hours_predictions(selected_dept):
    # Initialize predictor
    worker_predictor = WorkerMonteCarloPredictor()

    # Get historical data
    historical_data = raw_data.copy()
    historical_data["date"] = pd.to_datetime(historical_data["date"])

    # Get last 60 days of data
    last_date = historical_data["date"].max()
    start_date = last_date - pd.Timedelta(days=60)
    historical_data = historical_data[historical_data["date"] >= start_date]

    # Calculate daily average hours per worker
    daily_averages = (
        historical_data.groupby("date")
        .agg({"total_hours_charged": "sum", "userid": "nunique"})
        .reset_index()
    )

    daily_averages["avg_hours_per_worker"] = (
        daily_averages["total_hours_charged"] / daily_averages["userid"]
    )

    # Get worker predictions
    worker_predictions = worker_predictor.predict_all_workers(raw_data)
    next_dates = pd.date_range(start=last_date + timedelta(days=1), periods=5, freq="B")

    # Calculate average predicted hours per worker
    num_workers = len(worker_predictions)
    total_mean_prediction = np.zeros(5)
    total_lower_bound = np.zeros(5)
    total_upper_bound = np.zeros(5)

    for worker_id, prediction in worker_predictions.items():
        total_mean_prediction += prediction["mean_prediction"]
        total_lower_bound += prediction["lower_bound"]
        total_upper_bound += prediction["upper_bound"]

    avg_mean_prediction = total_mean_prediction / num_workers
    avg_lower_bound = total_lower_bound / num_workers
    avg_upper_bound = total_upper_bound / num_workers

    # Create figure
    fig = go.Figure()

    # Add historical average hours
    fig.add_trace(
        go.Scatter(
            x=daily_averages["date"],
            y=daily_averages["avg_hours_per_worker"],
            name="Historical Average Hours",
            line=dict(color="#2ecc71"),
        )
    )

    # Add predicted average hours
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=avg_mean_prediction,
            name="Predicted Average Hours",
            line=dict(color="#3498db", dash="dash"),
        )
    )

    # Add confidence interval
    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=avg_upper_bound,
            mode="lines",
            name="95% CI Upper",
            line=dict(width=0),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=next_dates,
            y=avg_lower_bound,
            mode="lines",
            name="95% CI Lower",
            fill="tonexty",
            fillcolor="rgba(52, 152, 219, 0.2)",
            line=dict(width=0),
            showlegend=False,
        )
    )

    # Add reference line for 8 hours
    fig.add_hline(
        y=8,
        line_dash="dash",
        line_color="red",
        annotation_text="8 Hour Standard",
        annotation_position="right",
    )

    # Add vertical line separating historical and predicted data
    fig.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(
            color="gray",
            width=1,
            dash="dash",
        ),
    )

    # Add annotation for the vertical line
    fig.add_annotation(
        x=last_date,
        y=1,
        yref="paper",
        text="Prediction Start",
        showarrow=False,
        textangle=-90,
        xanchor="left",
        yanchor="bottom",
    )

    # Calculate summary statistics
    overall_avg_hours = daily_averages["avg_hours_per_worker"].mean()
    days_over_eight = len(daily_averages[daily_averages["avg_hours_per_worker"] > 8])

    fig.update_layout(
        title=f"Average Hours per Worker<br>"
        f"Overall Avg: {overall_avg_hours:.1f} hrs | Days Over 8hrs: {days_over_eight}",
        xaxis_title="Date",
        yaxis_title="Average Hours per Worker",
        height=500,
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Set y-axis range with some padding
    y_max = max(
        daily_averages["avg_hours_per_worker"].max(),
        avg_upper_bound.max(),
        8.5,  # Ensure 8-hour line is visible
    )
    fig.update_yaxes(range=[0, y_max * 1.1])

    return fig


# Update hybrid forecast callback
@app.callback(
    Output("hybrid-forecast-chart", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
        Input("forecast-horizon", "value"),
        Input("forecast-options", "value"),
    ],
)
def update_hybrid_forecast(
    start_date, end_date, selected_dept, forecast_horizon, forecast_options
):

    try:
        filtered_data, filtered_raw = filter_data(start_date, end_date, selected_dept)

        if filtered_raw.empty:
            return create_empty_figure("No data available for selected filters")

        hybrid_predictor = HybridLaborPredictor()
        include_seasonality = (
            "seasonality" in forecast_options if forecast_options else False
        )
        include_growth = False

        mean_forecast, lower_bound, upper_bound = (
            hybrid_predictor.generate_hybrid_forecast(
                filtered_raw,
                forecast_horizon,
                include_seasonality=include_seasonality,
                include_growth=False,  # Always get base forecast
            )
        )

        # Calculate growth rate once
        growth_rate = hybrid_predictor.calculate_employee_growth_rate(filtered_raw)

        # Create figure
        fig = go.Figure()

        last_date = pd.to_datetime(filtered_raw["date"].max())
        forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1)[
            1:
        ]

        # Structure customdata as arrays
        historical_customdata = np.column_stack(
            (
                filtered_data["total_hours_charged"].values,
                np.full(len(filtered_data), growth_rate),
            )
        )

        forecast_customdata = np.column_stack(
            (mean_forecast, np.full(len(mean_forecast), growth_rate))
        )

        upper_customdata = np.column_stack(
            (upper_bound, np.full(len(upper_bound), growth_rate))
        )

        lower_customdata = np.column_stack(
            (lower_bound, np.full(len(lower_bound), growth_rate))
        )

        # Add traces with properly structured customdata
        fig.add_trace(
            go.Scatter(
                x=filtered_data["date"],
                y=filtered_data["total_hours_charged"],
                name="Historical",
                line=dict(color="#2ecc71"),
                customdata=historical_customdata,
                hovertemplate="Date: %{x}<br>Hours: %{y:.1f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=mean_forecast,
                name="Hybrid Forecast",
                line=dict(color="#e74c3c", dash="dash"),
                customdata=forecast_customdata,
                hovertemplate="Date: %{x}<br>Forecast: %{y:.1f}<extra></extra>",
            )
        )

        # Add confidence interval traces
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=upper_bound,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                customdata=upper_customdata,
                hovertemplate="Date: %{x}<br>Upper Bound: %{y:.1f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=lower_bound,
                mode="lines",
                fillcolor="rgba(231, 76, 60, 0.2)",
                fill="tonexty",
                showlegend=False,
                customdata=lower_customdata,
                hovertemplate="Date: %{x}<br>Lower Bound: %{y:.1f}<extra></extra>",
            )
        )

        # Add historical average reference line
        hist_avg = filtered_data["total_hours_charged"].mean()
        fig.add_hline(
            y=hist_avg,
            line_dash="dot",
            line_color="gray",
            annotation_text="Historical Average",
        )

        # Update the title to show actual growth state from forecast_options
        title = (
            f"Hybrid Labor Hours Forecast<br>"
            f"Seasonality: {'On' if include_seasonality else 'Off'} | "
            f"Growth: {'On' if 'growth' in forecast_options else 'Off'}"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Hours",
            height=400,
            template="plotly_white",
            hovermode="x unified",
            showlegend=True,
        )

        return fig

    except Exception as e:
        print(f"Error in update_hybrid_forecast: {str(e)}")  # Add debugging
        return create_empty_figure(f"Error: {str(e)}")


def create_empty_figure(message):
    return go.Figure().update_layout(
        title=message, xaxis=dict(showgrid=False), yaxis=dict(showgrid=False)
    )


# Helper function to filter data
def filter_data(start_date, end_date, selected_dept):
    """Filter data for both aggregated and raw data views.
    Returns a tuple of (filtered_agg, filtered_raw)"""

    # Filter raw data
    mask_raw = (raw_data["date"] >= start_date) & (raw_data["date"] <= end_date)
    if selected_dept != "All":
        mask_raw &= raw_data["dept"] == selected_dept
    filtered_raw = raw_data[mask_raw].copy()

    # Filter aggregated data
    if selected_dept == "All":
        mask_agg = (daily_data["date"] >= start_date) & (daily_data["date"] <= end_date)
        filtered_agg = daily_data[mask_agg].copy()
    else:
        mask_agg = (
            (dept_daily_data["date"] >= start_date)
            & (dept_daily_data["date"] <= end_date)
            & (dept_daily_data["dept"] == selected_dept)
        )
        filtered_agg = dept_daily_data[mask_agg].copy()

    # Add fiscal year calculations to both datasets
    for df in [filtered_agg, filtered_raw]:
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df["fiscal_year"] = df["date"].apply(
                lambda x: x.year if x.month >= 10 else x.year - 1
            )
            df["fiscal_period"] = df["date"].apply(lambda x: (x.month - 10) % 12 + 1)

    return filtered_agg, filtered_raw


# Add CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <script>
            window.dash_clientside = Object.assign({}, window.dash_clientside, {
                clientside: {
                    toggle_growth: function(options, figure) {
                        if (!figure || !figure.data) return figure;
                        
                        const includeGrowth = options.includes('growth');
                        const data = figure.data;
                        
                        data.forEach(trace => {
                            if (trace.customdata) {
                                const originalY = trace.customdata.map(d => d[0]);
                                const growthRate = trace.customdata[0][1];  // Same for all points
                                
                                if (includeGrowth) {
                                    trace.y = originalY.map((y, i) => 
                                        y * (1 + growthRate) ** (i/30)
                                    );
                                } else {
                                    trace.y = originalY;
                                }
                            }
                        });
                        
                        return {...figure, data};
                    }
                }
            });
        </script>
        <style>
            .kpi-card {
                backgroundColor: #f8f9fa;
                borderRadius: 10px;
                padding: 20px;
                textAlign: center;
                boxShadow: 0 2px 4px rgba(0,0,0,0.1);
                width: 250px;
            }
            .kpi-card h4 {
                margin: 0;
                color: #666;
            }
            .kpi-card h2 {
                margin: 10px 0 0 0;
                color: #333;
            }
            .metric-card {
                backgroundColor: #f8f9fa;
                borderRadius: 8px;
                padding: 15px;
                width: 30%;
                boxShadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .metric-card h5 {
                margin: 0 0 10px 0;
                color: #495057;
                fontSize: 1.1em;
            }
            .metric-content {
                fontSize: 1.2em;
                color: #212529;
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
