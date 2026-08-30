import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager

# Configure fonts globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

# --- 1. Generate updated table ---
fig, ax = plt.subplots(figsize=(14.5, 4.5))
ax.axis('tight')
ax.axis('off')

data = [
    ["McMahan et al. (2017)", "FedAvg Algorithm", "Base FL", "Base", "Ensures basic decentralization privacy", "Vulnerable to non-IID client drift"],
    ["Veličković et al. (2018)", "Graph Attention Networks", "98.1%", "N/A", "Extracts strong relational topologies", "Fails strict edge confidentiality rules"],
    ["Li et al. (2020)", "FedProx Regularization", "Robust", "Robust", "Stabilizes non-IID learning environments", "Lacks structural graph processing"],
    ["Zheng et al. (2021)", "Federated Subgraph Networks", "92.5%", "91.2%", "Enables distributed graph sub-training", "High communication & alignment cost"],
    ["Su et al. (2022)", "DP-Enabled Spatiotemporal GNN", "88.4%", "87.1%", "Hard privacy guarantees via noise limits", "Creates major utility-privacy tradeoff"],
    ["Wang et al. (2023)", "FedGraph & Edge Sampling", "94.2%", "93.0%", "Edge compute efficiency via sampling", "High complexity in node alignment"],
    ["Proposed Project", "Temporal GNN+FedProx+DP+HE", "99.1%", "97.5%", "High precision, absolute privacy layer", "Initial local graph compute load"]
]
columns = ["Literature Source", "Core Architecture", "Accuracy", "F1-Score", "Key Innovation & Advantage", "Major Limitation / Gap"]

table = ax.table(cellText=data, colLabels=columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11) # Requested size 11
table.scale(1.2, 2.2)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#aaaaaa')
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#dce6f1')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

plt.title('Table 1: Analytical Comparison of Relevant Literature', pad=30, fontsize=12, weight='bold')
plt.savefig('/Volumes/Untitled/federated-fraud-detection/lit_survey_table.png', bbox_inches='tight', dpi=300)
plt.close()

# --- 2. Generate Literature Summary Timeline ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(-3, 3)
ax.axis('off')

# Central timeline
ax.plot([1, 11], [0, 0], color='#2c3e50', linewidth=3, zorder=1)

events = [
    {"year": 2017, "title": "FedAvg Framework", "desc": "Baseline decentralized model\ntraining via averaging.", "pos": 2, "dir": 1},
    {"year": 2018, "title": "Graph Attention", "desc": "Relational structural feature\nextraction on graphs.", "pos": 3.4, "dir": -1},
    {"year": 2020, "title": "FedProx Setup", "desc": "Tackles non-IID financial\ndata distributions.", "pos": 4.8, "dir": 1},
    {"year": 2021, "title": "Federated Subgraphs", "desc": "First integration of decentralized\nlearning & GNNs.", "pos": 6.2, "dir": -1},
    {"year": 2022, "title": "DP-Enabled GNNs", "desc": "Noise injection into spatial\ncomponents for privacy.", "pos": 7.6, "dir": 1},
    {"year": 2023, "title": "FedGraph + Sampling", "desc": "Scalability and contrastive\nlearning for edge devices.", "pos": 9, "dir": -1},
    {"year": "Proposed", "title": "Temporal FL-GNN", "desc": "Multi-modal temporal graphs\nwith Multi-Layer Security.", "pos": 10.4, "dir": 1, "color": "#e74c3c"}
]

for idx, ev in enumerate(events):
    x = ev["pos"]
    y_dir = ev["dir"]
    color = ev.get("color", "#2980b9")
    
    # Draw points on timeline
    ax.scatter(x, 0, color=color, s=150, zorder=5, edgecolors='white', linewidths=2)
    
    # Draw stems
    ax.plot([x, x], [0, 1.2 * y_dir], color='gray', linestyle='--', linewidth=1.5, zorder=2)
    
    # Text box logic
    box_y = 1.6 * y_dir if y_dir == 1 else 1.8 * y_dir
    
    # Year text
    ax.text(x, -0.4 * y_dir, str(ev["year"]), ha='center', va='center', fontsize=11, weight='bold', color='#34495e')
    
    # Description box
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec=color, lw=2)
    text_str = f"{ev['title']}\n\n{ev['desc']}"
    
    ax.text(x, box_y, text_str, ha="center", va="center", size=11,
            bbox=bbox_props, color='#2c3e50')

plt.title("Figure 1: Evolution of Federated Graph Networks (Literature Summary)", size=14, weight='bold', y=0.95)
plt.savefig('/Volumes/Untitled/federated-fraud-detection/lit_survey_timeline.png', bbox_inches='tight', dpi=300)
plt.close()

print("Images generated successfully")
