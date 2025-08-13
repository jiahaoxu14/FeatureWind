#!/bin/bash

# Start FeatureWind backend server
echo "Starting FeatureWind backend server..."

cd backend

# Install dependencies if needed
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Start Flask server
echo "Starting Flask server on http://localhost:5000"
python app.py