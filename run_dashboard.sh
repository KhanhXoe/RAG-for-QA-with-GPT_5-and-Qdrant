#!/bin/bash

# Run the visualization dashboard
echo "Starting RAG Pipeline Monitor..."
echo "================================"
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Streamlit not found. Installing..."
    pip install streamlit plotly pandas
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run the app
cd src/logging/viz
streamlit run app.py --server.port 8501 --server.address 0.0.0.0

echo ""
echo "Dashboard stopped."
