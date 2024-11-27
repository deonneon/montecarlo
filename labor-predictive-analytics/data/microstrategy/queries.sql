-- labor_data

                SELECT 
                    date, userid, dept,
                    total_hours_charged, direct_hours,
                    non_direct_hours, overtime_hours,
                    fiscal_year, fiscal_period
                FROM labor_data
                WHERE date BETWEEN @start_date AND @end_date
            

-- department_metrics

                SELECT 
                    dept, date,
                    SUM(total_hours_charged) as total_hours,
                    SUM(direct_hours) as direct_hours,
                    SUM(overtime_hours) as overtime_hours,
                    COUNT(DISTINCT userid) as employee_count
                FROM labor_data
                GROUP BY dept, date
            

-- worker_metrics

                SELECT 
                    userid, date,
                    total_hours_charged,
                    direct_hours,
                    overtime_hours
                FROM labor_data
                WHERE userid = @worker_id
            

-- fiscal_metrics

                SELECT 
                    fiscal_year,
                    fiscal_period,
                    SUM(total_hours_charged) as total_hours,
                    AVG(total_hours_charged) as avg_hours,
                    COUNT(DISTINCT userid) as employee_count
                FROM labor_data
                GROUP BY fiscal_year, fiscal_period
            

