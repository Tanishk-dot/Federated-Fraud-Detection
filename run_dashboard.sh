#!/bin/bash

# Federated Fraud Detection Dashboard Launcher

echo "🚀 Starting Federated Fraud Detection Dashboard..."
echo "=================================================="
echo ""
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo ""

cd dashboard
streamlit run app.py --server.port 8501 --server.address localhost
