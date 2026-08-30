"""
Generate a professional Gantt chart — Unified Future Roadmap
Monthly timeline: May 2026 → March 2027
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.facecolor': '#0f1117',
    'figure.facecolor': '#0f1117',
    'text.color': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'xtick.color': '#a0a0a0',
    'ytick.color': '#a0a0a0',
})

# Months on the x-axis: May 2026 (index 0) → Mar 2027 (index 10)
month_labels = [
    "May '26", "Jun '26", "Jul '26", "Aug '26", "Sep '26", "Oct '26",
    "Nov '26", "Dec '26", "Jan '27", "Feb '27", "Mar '27",
]
num_months = len(month_labels)  # 11

# ── Tasks: (label, start_month_index, duration_months) ───────────
tasks = [
    ("Hyperparameter Tuning (LR, Rounds)",         0,  3),
    ("Adaptive DP Budget Scheduling",              1,  3),
    ("Cross-Silo Scalability Testing",             2,  2),
    ("FedProx to FedBN Strategy Migration",        3,  2),
    ("HE-Accelerated Aggregation Pipeline",        2,  3),
    ("Adversarial Robustness Evaluation",          4,  2),
    ("Containerized Deployment (Docker/K8s)",      6,  2),
    ("Real-Time Inference API (FastAPI)",           6,  3),
    ("Cross-Dataset Generalization Benchmark",     7,  2),
    ("Comprehensive Ablation Studies",             8,  2),
    ("Final Dashboard Polish & Demo Prep",         9,  2),
    ("Research Paper & Documentation",             8,  3),
]

# Smooth purple → teal gradient
bar_colors = [
    "#6C63FF", "#7B6FFF", "#8A7BFF", "#9987FF",
    "#A893FF", "#B79FFF", "#00C9A7", "#1AD4B5",
    "#33DFC3", "#4DEAD1", "#66F5DF", "#80FFED",
]

# ── Build figure ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 8.5))

num_tasks = len(tasks)
bar_height = 0.55
y_positions = list(range(num_tasks - 1, -1, -1))

# Subtle horizontal gridlines
for y in y_positions:
    ax.axhline(y=y, color='#1a1d2e', linewidth=0.5, zorder=0)

# Subtle vertical month gridlines (at each month boundary)
for m in range(num_months):
    ax.axvline(x=m, color='#1a1d2e', linewidth=0.4, linestyle='--', zorder=0)

# ── Draw bars ────────────────────────────────────────────────────
for i, (label, start, duration) in enumerate(tasks):
    y = y_positions[i]
    color = bar_colors[i]

    bar = FancyBboxPatch(
        (start + 0.05, y - bar_height / 2),
        duration - 0.1,
        bar_height,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='white',
        linewidth=0.6,
        alpha=0.85,
        zorder=3,
    )
    ax.add_patch(bar)

# ── Y-axis labels ────────────────────────────────────────────────
task_labels = [t[0] for t in tasks]
ax.set_yticks(y_positions)
ax.set_yticklabels(task_labels, fontsize=10.5)

# ── X-axis (months, equally spaced) ─────────────────────────────
ax.set_xticks([m + 0.5 for m in range(num_months)])
ax.set_xticklabels(month_labels, fontsize=10, rotation=30, ha='right')
ax.set_xlim(-0.2, num_months + 0.2)
ax.set_ylim(-0.8, num_tasks + 0.5)

# ── Title ────────────────────────────────────────────────────────
ax.set_title(
    "Federated Fraud Detection — Future Roadmap",
    fontsize=18, fontweight='bold', color='white', pad=22,
)

# ── Cleanup spines ───────────────────────────────────────────────
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis='both', length=0)

plt.tight_layout()
plt.savefig("gantt_phase3_4.png", dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print("Done: gantt_phase3_4.png")
