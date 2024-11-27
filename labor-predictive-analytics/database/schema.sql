-- Labor data table
CREATE TABLE labor_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    userid VARCHAR(10) NOT NULL,
    dept VARCHAR(50) NOT NULL,
    total_hours_charged DECIMAL(10,2) NOT NULL,
    direct_hours DECIMAL(10,2) NOT NULL,
    non_direct_hours DECIMAL(10,2) NOT NULL,
    overtime_hours DECIMAL(10,2) NOT NULL,
    is_holiday BOOLEAN NOT NULL,
    fiscal_year INT NOT NULL,
    fiscal_period INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Department metrics table
CREATE TABLE department_metrics (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    dept VARCHAR(50) NOT NULL,
    total_hours DECIMAL(10,2) NOT NULL,
    direct_hours DECIMAL(10,2) NOT NULL,
    overtime_hours DECIMAL(10,2) NOT NULL,
    employee_count INT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Forecast results table
CREATE TABLE forecast_results (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    forecast_type VARCHAR(50) NOT NULL,
    mean_value DECIMAL(10,2) NOT NULL,
    lower_bound DECIMAL(10,2) NOT NULL,
    upper_bound DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);