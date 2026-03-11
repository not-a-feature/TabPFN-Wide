import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Match the project's plot style
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Colors from MODEL_CONFIG in plot_results.py
MODEL_CONFIG = {
    "TabPFN-Wide (5k)": "#2ca02c",
    "Random Forest": "#5D4037",
    "RealMLP": "#8c564b",
    "XGBoost": "#e377c2",
    "TabPFN v2": "#1f77b4",
    "TabICL": "#9467bd",
}

# Data from the image (same order)
methods = ["TabPFN-Wide (5k)", "Random Forest", "RealMLP", "XGBoost", "TabPFN v2", "TabICL"]
win_rates = [92.0, 68.0, 56.0, 44.0, 34.7, 5.3]
colors = [MODEL_CONFIG[m] for m in methods]

fig, ax = plt.subplots(figsize=(8, 5))

x_pos = np.arange(len(methods))
bars = ax.bar(x_pos, win_rates, color=colors, edgecolor="black", linewidth=0.8, width=0.7)

# Add percentage labels on top of each bar
for bar, val in zip(bars, win_rates):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.5,
        f"{val}%",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

ax.set_xticks(x_pos)
ax.set_xticklabels(methods, rotation=45, ha="right")
ax.set_ylabel("Win Rate (%)")
ax.set_xlabel("Method")
ax.set_ylim(0, 105)

plt.tight_layout()
fig.savefig("win_rate_bar.pdf", bbox_inches="tight", dpi=300)
print("Saved: win_rate_bar.pdf")
plt.close(fig)
