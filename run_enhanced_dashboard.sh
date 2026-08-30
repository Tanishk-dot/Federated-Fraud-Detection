#!/bin/bash

# Enhanced Dashboard Launch Script
echo "🚀 Launching Enhanced Federated Fraud Detection Dashboard..."
echo ""
echo "Features:"
echo "  ✅ How It Works - Animated FL process"
echo "  ✅ Privacy Guarantees - DP demonstrations"
echo "  ✅ Graph Intelligence - Fraud ring detection"
echo "  ✅ Contrastive Learning - Novel fraud detection"
echo "  ✅ Live Inference - Interactive fraud analysis"
echo "  ✅ Comparison - Why this approach wins"
echo ""

cd dashboard
streamlit run app_enhanced.py --server.port 8501 --server.headless true

echo ""
echo "Dashboard stopped."
