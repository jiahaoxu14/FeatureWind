#!/bin/bash

# Start FeatureWind frontend server
echo "Starting FeatureWind frontend server..."

cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing Node.js dependencies..."
    npm install
fi

# Start React development server
echo "Starting React development server on http://localhost:3000"
npm start