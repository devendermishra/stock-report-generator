#!/bin/bash
# Simple script to run the stock report generator

# Activate virtual environment
source venv/bin/activate

# Run the command as a module to handle imports properly
python -m src.main "$@"

