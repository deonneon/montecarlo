import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from data_processing import DataPreprocessor
from time_series_model import TimeSeriesPredictor
from monte_carlo import MonteCarloSimulator

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
                "margin-bottom": "30px",
                "padding": "20px",
                "background-color": "#f8f9fa",
                "border-radius": "10px",
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
                                "margin-bottom": "10px",
                                "font-weight": "bold",
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
                        "margin-right": "20px",
                        "vertical-align": "top",
                    },
                ),
                # Department Filter
                html.Div(
                    [
                        html.Label(
                            "Department:",
                            style={
                                "margin-bottom": "10px",
                                "font-weight": "bold",
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
                        "vertical-align": "top",
                    },
                ),
            ],
            style={
                "margin": "20px",
                "padding": "20px",
                "background-color": "#f8f9fa",
                "border-radius": "10px",
                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
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
                "justify-content": "space-around",
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
                        "margin-bottom": "20px",
                        "background-color": "#fff",
                        "border-radius": "10px",
                        "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
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
                                "background-color": "#fff",
                                "border-radius": "10px",
                                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "padding": "15px",
                                "margin-right": "2%",
                            },
                        ),
                        # Forecast Chart
                        html.Div(
                            dcc.Graph(id="forecast"),
                            style={
                                "width": "48%",
                                "display": "inline-block",
                                "background-color": "#fff",
                                "border-radius": "10px",
                                "box-shadow": "0 2px 4px rgba(0,0,0,0.1)",
                                "padding": "15px",
                            },
                        ),
                    ],
                    style={"display": "flex", "justify-content": "space-between"},
                ),
            ],
            style={"margin": "20px", "padding": "10px"},
        ),
    ],
    style={"background-color": "#f0f2f5", "min-height": "100vh", "padding": "20px"},
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
    filtered_data = filter_data(start_date, end_date, selected_dept)

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
    filtered_data = filter_data(start_date, end_date, selected_dept)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_data["date"],
            y=filtered_data["total_hours_charged"],
            mode="lines",
            name="Total Hours",
        )
    )

    title = (
        "Labor Hours Trend - All Departments"
        if selected_dept == "All"
        else f"Labor Hours Trend - {selected_dept}"
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Hours",
        height=400,
        template="plotly_white",
    )

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


# Forecast callback
@app.callback(
    Output("forecast", "figure"),
    [
        Input("date-picker", "start_date"),
        Input("date-picker", "end_date"),
        Input("department-filter", "value"),
    ],
)
def update_forecast(start_date, end_date, selected_dept):
    # Filter data
    filtered_data = filter_data(start_date, end_date, selected_dept)

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


# Helper function to filter data
def filter_data(start_date, end_date, selected_dept):
    if selected_dept == "All":
        mask = (daily_data["date"] >= start_date) & (daily_data["date"] <= end_date)
        return daily_data[mask].copy()
    else:
        mask = (
            (dept_daily_data["date"] >= start_date)
            & (dept_daily_data["date"] <= end_date)
            & (dept_daily_data["dept"] == selected_dept)
        )
        return dept_daily_data[mask].copy()


# Add CSS styling
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .kpi-card {
                background-color: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
