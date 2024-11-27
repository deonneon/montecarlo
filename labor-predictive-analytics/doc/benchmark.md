# Benchmark

When analyzing this benchmark results, you should look for several key aspects:

Accuracy Metrics:

MAPE (Mean Absolute Percentage Error): Lower is better, tells you the average percentage error
RMSE (Root Mean Square Error): Lower is better, penalizes large errors more heavily
MAE (Mean Absolute Error): Lower is better, shows average absolute difference in hours

Comparative Performance:

Which model performs best overall? (Aggregate vs Worker vs Hybrid)
Are there significant differences between the models, or are they relatively close?
Does the Hybrid model successfully combine the strengths of both approaches?

Patterns to Watch:

How well does each model handle:

Seasonal patterns (like end-of-month or end-of-quarter spikes)
Holiday periods
Overtime periods
Department-specific variations

Practical Considerations:

Is the most accurate model significantly better to justify its complexity?
Are the error rates acceptable for practical labor planning?
A MAPE under 10% is often considered good for labor forecasting
Consider if the errors would lead to over or understaffing
