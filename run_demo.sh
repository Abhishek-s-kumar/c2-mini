#!/bin/bash

# C2 Beacon Detection - Complete Demo Runner

echo "========================================"
echo "C2 Beacon Detection Demo"
echo "========================================"

# Activate virtual environment
source venv/bin/activate

# Step 1: Time series extraction
echo ""
echo "STEP 1: Time Series Extraction"
echo "--------------------------------"
python3 time_series_extractor.py

# Step 2: FFT Analysis
echo ""
echo "STEP 2: FFT Analysis"
echo "--------------------------------"
python3 fft_analyzer.py

# Step 3: Show results
echo ""
echo "========================================"
echo "DEMO COMPLETE!"
echo "========================================"
echo ""
echo "Files generated:"
ls -la *.csv *.png *.json 2>/dev/null | awk '{print $NF}'
echo ""
echo "To show your professor:"
echo "1. time_series_extraction_demo.png - Shows log → time series"
echo "2. fft_analysis_demo.png - Shows FFT and P-score calculation"
echo "3. fft_results.json - Contains detection results"
echo ""
echo "Quick commands to run during meeting:"
echo "  python3 time_series_extractor.py"
echo "  python3 fft_analyzer.py"
echo "========================================"
