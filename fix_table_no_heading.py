import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager

# Configure fonts globally
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']

# --- 1. Generate updated table ---
# Increased figure width to prevent overlap
fig, ax = plt.subplots(figsize=(18, 5)) 
ax.axis('tight')
ax.axis('off')

# Modified text to include manual line breaks where it might be too long
data = [
    ["McMahan et al.\n(2017)", "FedAvg Algorithm", "Base FL", "Base", "Ensures basic decentralization privacy", "Vulnerable to non-IID client drift"],
    ["Veličković et al.\n(2018)", "Graph Attention Networks", "98.1%", "N/A", "Extracts strong relational topologies", "Fails strict edge confidentiality rules"],
    ["Li et al.\n(2020)", "FedProx Regularization", "Robust", "Robust", "Stabilizes non-IID\nlearning environments", "Lacks structural graph processing"],
    ["Zheng et al.\n(2021)", "Federated Subgraph\nNetworks", "92.5%", "91.2%", "Enables distributed graph sub-training", "High communication &\nalignment cost"],
    ["Su et al.\n(2022)", "DP-Enabled\nSpatiotemporal GNN", "88.4%", "87.1%", "Hard privacy guarantees via noise limits", "Creates major utility-privacy\ntradeoff"],
    ["Wang et al.\n(2023)", "FedGraph & Edge\nSampling", "94.2%", "93.0%", "Edge compute efficiency via sampling", "High complexity in node alignment"],
    ["Proposed Project", "Temporal GNN + FedProx\n+ DP + HE", "99.1%", "97.5%", "High precision, absolute privacy layer", "Initial local graph compute load"]
]
columns = ["Literature Source", "Core Architecture", "Accuracy", "F1-Score", "Key Innovation & Advantage", "Major Limitation / Gap"]

# Define explicit column widths so they don't squash
col_widths = [0.12, 0.16, 0.09, 0.09, 0.26, 0.28]

table = ax.table(cellText=data, colLabels=columns, colWidths=col_widths, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(11) 
table.scale(1, 4) # Adjust height scaling

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#aaaaaa')
    
    # Add some padding to cells
    cell.PAD = 0.05
    
    if row == 0:
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#dce6f1')
    else:
        if row % 2 == 0:
            cell.set_facecolor('#f2f2f2')

# REMOVED: plt.title(...)
plt.savefig('/Volumes/Untitled/federated-fraud-detection/lit_survey_table.png', bbox_inches='tight', dpi=300)
plt.close()

print("Table image successfully regenerated without heading")
