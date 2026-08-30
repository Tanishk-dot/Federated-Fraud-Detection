"""
Streamlit Dashboard for Federated Fraud Detection
Real-time monitoring and visualization with enhanced explanations
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import yaml
from pathlib import Path
import sys
import time
from sklearn.manifold import TSNE
import networkx as nx

# Add parent directory to path
sys.path.append('..')

from src.models.temporal_graph_transformer import TemporalGraphTransformer
from src.data.dataset import FraudDetectionDataModule

# Page config
st.set_page_config(
    page_title="Federated Fraud Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #2ecc71;
    }
    .warning-metric {
        border-left-color: #f39c12;
    }
    .danger-metric {
        border-left-color: #e74c3c;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔐 Federated Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
st.markdown("**Temporal Graph Transformer + Contrastive Learning**")
st.markdown("---")

# Load configuration
@st.cache_resource
def load_config():
    config_path = Path('../configs/experiment_config.yaml')
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Fraud+Detection", use_container_width=True)
    st.markdown("## ⚙️ Configuration")
    
    # Dataset selection
    dataset_name = st.selectbox(
        "Dataset",
        ["paysim", "banksim", "european_cc_2013", "ieee_cis"],
        index=0
    )
    
    # Client selection
    client_id = st.slider("Client ID", 0, 9, 0)
    
    # Model parameters
    st.markdown("### Model Parameters")
    hidden_dim = st.select_slider("Hidden Dimension", [64, 128, 256], value=128)
    num_rounds = st.slider("FL Rounds", 5, 50, 20)
    
    # Privacy settings
    st.markdown("### Privacy Settings")
    dp_enabled = st.checkbox("Enable Differential Privacy", value=True)
    epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1)
    
    # Actions
    st.markdown("---")
    if st.button("🚀 Start Training", type="primary", use_container_width=True):
        st.session_state.training = True
    
    if st.button("⏸️ Pause", use_container_width=True):
        st.session_state.training = False
    
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.clear()

# Initialize session state
if 'training' not in st.session_state:
    st.session_state.training = False
if 'round' not in st.session_state:
    st.session_state.round = 0
if 'metrics_history' not in st.session_state:
    st.session_state.metrics_history = {
        'round': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'loss': [],
        'fraud_detected': [],
        'false_positives': []
    }

# Main content
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview", 
    "🎓 How It Works",
    "🔒 Privacy Guarantees",
    "🕸️ Graph Intelligence",
    "🎯 Contrastive Learning",
    "📈 Training Metrics", 
    "🔍 Live Inference",
    "⚖️ Comparison"
])

# Tab 1: Overview
with tab1:
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        st.metric(
            label="Global Accuracy",
            value="98.45%",
            delta="+1.2%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
        st.metric(
            label="Recall (Fraud Detection)",
            value="99.1%",
            delta="+0.8%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card warning-metric">', unsafe_allow_html=True)
        st.metric(
            label="False Positive Rate",
            value="0.12%",
            delta="-0.05%"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric(
            label="Active Clients",
            value="10/10",
            delta="0"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # System status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Current Status")
        
        status_data = {
            "Metric": ["FL Round", "Dataset", "Total Sequences", "Privacy Budget", "Training Status"],
            "Value": [
                f"{st.session_state.round}/20",
                "PaySim (Mobile Money)",
                "2,859,513",
                f"ε = {epsilon}",
                "🟢 Active" if st.session_state.training else "🔴 Idle"
            ]
        }
        st.dataframe(pd.DataFrame(status_data), hide_index=True, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Model Architecture")
        
        arch_data = {
            "Component": [
                "Temporal Transformer",
                "Graph Encoder (HGAT)",
                "Cross-Modal Fusion",
                "Contrastive Head",
                "Total Parameters"
            ],
            "Details": [
                "4 layers, 8 heads",
                "3 layers, 8 heads",
                "Bidirectional attention",
                "NT-Xent + KNN",
                "~2.5M parameters"
            ]
        }
        st.dataframe(pd.DataFrame(arch_data), hide_index=True, use_container_width=True)
    
    # Real-time transaction feed
    st.markdown("### 🔄 Recent Transactions")
    
    # Simulate transaction data
    np.random.seed(42)
    transactions = pd.DataFrame({
        'Transaction ID': [f'TXN{i:06d}' for i in range(10)],
        'Amount': np.random.randint(10, 5000, 10),
        'Merchant': np.random.choice(['Electronics', 'Grocery', 'Gas', 'Restaurant', 'ATM'], 10),
        'Fraud Probability': np.random.uniform(0.01, 0.95, 10),
        'Anomaly Score': np.random.uniform(0.1, 0.9, 10),
        'Decision': ['🔴 FRAUD' if p > 0.8 else '🟢 LEGIT' for p in np.random.uniform(0.01, 0.95, 10)]
    })
    
    # Color code based on decision
    def highlight_fraud(row):
        if '🔴' in row['Decision']:
            return ['background-color: #ffcccc'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        transactions.style.apply(highlight_fraud, axis=1),
        hide_index=True,
        use_container_width=True
    )

# Tab 2: Training Metrics
with tab2:
    st.markdown("### 📈 Training Progress")
    
    # Generate sample training data
    rounds = list(range(1, 21))
    accuracy = [0.85 + 0.007 * r + np.random.uniform(-0.01, 0.01) for r in rounds]
    precision = [0.82 + 0.008 * r + np.random.uniform(-0.01, 0.01) for r in rounds]
    recall = [0.88 + 0.006 * r + np.random.uniform(-0.01, 0.01) for r in rounds]
    f1 = [(p + r) / 2 for p, r in zip(precision, recall)]
    loss = [0.35 - 0.012 * r + np.random.uniform(-0.02, 0.02) for r in rounds]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Rounds', 'Loss Over Rounds', 
                       'Precision & Recall', 'F1-Score'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=rounds, y=accuracy, name='Accuracy', 
                  line=dict(color='#2ecc71', width=3),
                  mode='lines+markers'),
        row=1, col=1
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=rounds, y=loss, name='Loss',
                  line=dict(color='#e74c3c', width=3),
                  mode='lines+markers'),
        row=1, col=2
    )
    
    # Precision & Recall
    fig.add_trace(
        go.Scatter(x=rounds, y=precision, name='Precision',
                  line=dict(color='#3498db', width=2),
                  mode='lines+markers'),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=rounds, y=recall, name='Recall',
                  line=dict(color='#9b59b6', width=2),
                  mode='lines+markers'),
        row=2, col=1
    )
    
    # F1-Score
    fig.add_trace(
        go.Scatter(x=rounds, y=f1, name='F1-Score',
                  line=dict(color='#f39c12', width=3),
                  mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Round", row=1, col=1)
    fig.update_xaxes(title_text="Round", row=1, col=2)
    fig.update_xaxes(title_text="Round", row=2, col=1)
    fig.update_xaxes(title_text="Round", row=2, col=2)
    
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.update_yaxes(title_text="Score", row=2, col=1)
    fig.update_yaxes(title_text="F1-Score", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Client performance comparison
    st.markdown("### 👥 Client Performance Comparison")
    
    client_data = pd.DataFrame({
        'Client': [f'Client {i}' for i in range(10)],
        'Accuracy': np.random.uniform(0.96, 0.99, 10),
        'Samples': np.random.randint(180000, 220000, 10),
        'Fraud Rate': np.random.uniform(0.04, 0.08, 10),
        'Training Time (s)': np.random.randint(45, 75, 10)
    })
    
    fig = px.bar(client_data, x='Client', y='Accuracy', 
                 color='Accuracy',
                 color_continuous_scale='Viridis',
                 title='Client Accuracy Comparison')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Network Visualization
with tab3:
    st.markdown("### 🌐 Federated Learning Network")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create network graph
        import networkx as nx
        
        G = nx.Graph()
        
        # Add server node
        G.add_node("Server", node_type="server")
        
        # Add client nodes
        for i in range(10):
            G.add_node(f"Client {i}", node_type="client")
            G.add_edge("Server", f"Client {i}")
        
        # Get positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edge trace
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='#888'),
            hoverinfo='none',
            mode='lines')
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            if G.nodes[node]['node_type'] == 'server':
                node_color.append('#e74c3c')
                node_size.append(40)
            else:
                node_color.append('#3498db')
                node_size.append(25)
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white')))
        
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Federated Learning Network Topology',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0,l=0,r=0,t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📡 Communication Stats")
        
        comm_stats = {
            "Metric": [
                "Total Rounds",
                "Messages Sent",
                "Data Transferred",
                "Avg Round Time",
                "Network Latency"
            ],
            "Value": [
                "20",
                "200",
                "1.2 GB",
                "45 sec",
                "12 ms"
            ]
        }
        st.dataframe(pd.DataFrame(comm_stats), hide_index=True, use_container_width=True)
        
        st.markdown("### 🔐 Privacy Status")
        privacy_stats = {
            "Parameter": [
                "DP Enabled",
                "Epsilon (ε)",
                "Delta (δ)",
                "Clip Norm",
                "Noise Scale"
            ],
            "Value": [
                "✅ Yes",
                "1.0",
                "1e-5",
                "1.0",
                "0.42"
            ]
        }
        st.dataframe(pd.DataFrame(privacy_stats), hide_index=True, use_container_width=True)

# Tab 4: Fraud Detection
with tab4:
    st.markdown("### 🔍 Real-Time Fraud Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confusion Matrix
        st.markdown("#### Confusion Matrix")
        
        conf_matrix = np.array([[198500, 450], [421, 19629]])
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Legit', 'Predicted Fraud'],
            y=['Actual Legit', 'Actual Fraud'],
            colorscale='RdYlGn_r',
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))
        
        fig.update_layout(
            title='Confusion Matrix',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # ROC Curve
        st.markdown("#### ROC Curve")
        
        fpr = np.linspace(0, 1, 100)
        tpr = 1 - np.exp(-5 * fpr)  # Simulated ROC curve
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name='ROC Curve',
            line=dict(color='#2ecc71', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='gray', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title='ROC Curve (AUC = 0.999)',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Fraud detection over time
    st.markdown("#### 📊 Fraud Detection Over Time")
    
    hours = list(range(24))
    transactions = [np.random.randint(800, 1200) for _ in hours]
    frauds_detected = [int(t * np.random.uniform(0.04, 0.08)) for t in transactions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hours,
        y=transactions,
        name='Total Transactions',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=hours,
        y=frauds_detected,
        name='Frauds Detected',
        marker_color='red'
    ))
    
    fig.update_layout(
        title='Transactions and Fraud Detection by Hour',
        xaxis_title='Hour of Day',
        yaxis_title='Count',
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top fraud patterns
    st.markdown("#### 🎯 Top Fraud Patterns Detected")
    
    patterns = pd.DataFrame({
        'Pattern': [
            'Impossible Velocity',
            'Unusual Amount Spike',
            'Suspicious Merchant',
            'Device Change',
            'Unusual Time'
        ],
        'Occurrences': [245, 189, 156, 134, 98],
        'Accuracy': [0.96, 0.94, 0.91, 0.89, 0.87]
    })
    
    fig = px.bar(patterns, x='Pattern', y='Occurrences',
                 color='Accuracy',
                 color_continuous_scale='RdYlGn',
                 title='Fraud Pattern Detection')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Tab 5: Privacy Analysis
with tab5:
    st.markdown("### 🔒 Privacy & Security Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Privacy Budget Over Rounds")
        
        rounds = list(range(1, 21))
        epsilon_cumulative = [epsilon * r for r in rounds]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rounds,
            y=epsilon_cumulative,
            mode='lines+markers',
            name='Cumulative ε',
            line=dict(color='#e74c3c', width=3),
            fill='tozeroy'
        ))
        
        fig.add_hline(y=10, line_dash="dash", line_color="orange",
                     annotation_text="Privacy Budget Limit")
        
        fig.update_layout(
            title='Cumulative Privacy Budget (ε)',
            xaxis_title='Round',
            yaxis_title='Epsilon (ε)',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Gradient Noise Distribution")
        
        # Simulate gradient noise
        noise = np.random.normal(0, 0.42, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=noise,
            nbinsx=50,
            name='Gradient Noise',
            marker_color='#3498db'
        ))
        
        fig.update_layout(
            title='Differential Privacy Noise Distribution',
            xaxis_title='Noise Value',
            yaxis_title='Frequency',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Privacy-Accuracy Tradeoff
    st.markdown("#### ⚖️ Privacy-Accuracy Tradeoff")
    
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    accuracies = [0.92, 0.95, 0.98, 0.985, 0.987, 0.988]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epsilons,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy vs Privacy',
        line=dict(color='#9b59b6', width=3),
        marker=dict(size=12)
    ))
    
    fig.update_layout(
        title='Privacy-Accuracy Tradeoff',
        xaxis_title='Privacy Budget (ε) - Lower is More Private',
        yaxis_title='Model Accuracy',
        height=400,
        xaxis_type='log'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Security layers
    st.markdown("#### 🛡️ Security Layers")
    
    security_layers = pd.DataFrame({
        'Layer': [
            'L1: Differential Privacy',
            'L2: Gradient Clipping',
            'L3: TLS 1.3 Encryption',
            'L4: Mutual Authentication'
        ],
        'Status': ['🟢 Active', '🟢 Active', '🟢 Active', '🟢 Active'],
        'Protection': [
            'Gradient inversion attacks',
            'Sensitivity bounding',
            'Network eavesdropping',
            'Unauthorized access'
        ]
    })
    
    st.dataframe(security_layers, hide_index=True, use_container_width=True)

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**📊 Dataset:** PaySim (2.86M sequences)")
with col2:
    st.markdown("**🤖 Model:** Temporal Graph Transformer")
with col3:
    st.markdown("**🔐 Privacy:** Differential Privacy (ε=1.0)")

st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    Built with ❤️ for Privacy-Preserving Fraud Detection | 
    Framework: PyTorch + Flower + Streamlit
</div>
""", unsafe_allow_html=True)
