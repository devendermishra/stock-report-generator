# Data Directory Structure

This directory contains all data files used by the Stock Report Generator.

## Directory Structure

- `inputs/` - Input data files (CSV, JSON, etc.)
- `outputs/` - Generated reports and analysis outputs
- `processed/` - Intermediate processed data files
- `raw/` - Raw data files from external sources

## Usage

- Place input data files in the `inputs/` directory
- Generated reports will be saved to the `outputs/` directory
- Intermediate processing files are stored in `processed/`
- Raw data from external APIs/sources goes in `raw/`

## File Naming Convention

- Use descriptive names with timestamps
- Format: `{type}_{symbol}_{date}_{time}.{extension}`
- Example: `stock_data_TCS_2024-01-15_143022.json`
