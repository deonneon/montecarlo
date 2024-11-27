CREATE TABLE forecast_results (
    Date DATE,
    Total_Hours DOUBLE,
    Deterministic_Forecast DOUBLE,
    Monte_Carlo_Forecast DOUBLE,
    Confidence_Interval_5 DOUBLE,
    Confidence_Interval_95 DOUBLE,
    Is_Historical BOOLEAN,
    Generation_Timestamp TIMESTAMP
)
PARTITIONED BY (Year INT, Month INT)
STORED AS PARQUET;