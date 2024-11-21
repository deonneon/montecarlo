#!/bin/bash

# Define project root directory
PROJECT_NAME="labor-predictive-analytics"
mkdir -p $PROJECT_NAME

# Create subdirectories
mkdir -p $PROJECT_NAME/{data/{raw,processed,generated},notebooks,src,tests}

# Create empty files in the root directory
touch $PROJECT_NAME/requirements.txt
touch $PROJECT_NAME/setup.py
touch $PROJECT_NAME/Dockerfile
touch $PROJECT_NAME/README.md
touch $PROJECT_NAME/.gitignore
touch $PROJECT_NAME/.env

# Create empty source files in the src directory
touch $PROJECT_NAME/src/__init__.py
touch $PROJECT_NAME/src/data_processing.py
touch $PROJECT_NAME/src/kpi_calculation.py
touch $PROJECT_NAME/src/monte_carlo.py
touch $PROJECT_NAME/src/dashboard.py
touch $PROJECT_NAME/src/utils.py

# Create empty test files in the tests directory
touch $PROJECT_NAME/tests/test_data_processing.py
touch $PROJECT_NAME/tests/test_kpi_calculation.py
touch $PROJECT_NAME/tests/test_monte_carlo.py
touch $PROJECT_NAME/tests/test_dashboard.py

# Create empty placeholder files in data subdirectories
touch $PROJECT_NAME/data/raw/.gitkeep
touch $PROJECT_NAME/data/processed/.gitkeep
touch $PROJECT_NAME/data/generated/.gitkeep

# Output completion message
echo "Project structure for '$PROJECT_NAME' has been created."
