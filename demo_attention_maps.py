import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabpfnwide.classifier import TabPFNWideClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target
feature_names = dataset.feature_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = X_train[:20]
y = y_train[:20]

print("Load model")

clf = TabPFNWideClassifier(
    model_name="wide-v2-5k",
    device="cpu",
    n_estimators=1,  # Only works with 1 estimator and no grouping.
    features_per_group=1,
    save_attention_maps=True,
)


print("Fitting and predicting...")
clf.fit(X, y)
proba = clf.predict_proba(X)

# ==============================================================================
# PART 1: Feature-to-Feature Attention Maps
# ==============================================================================
print("\n=== Feature-to-Feature Attention Maps ===")
maps = clf.get_attention_maps()

# Average across layers for a summary view
avg_map = np.mean(maps, axis=0)
print(f"Attention Map Shape: {avg_map.shape}")

plt.figure(figsize=(10, 8))
plt.imshow(avg_map, cmap="viridis", interpolation="nearest")
plt.colorbar(label="Attention Weight")
plt.title("Average Attention Map (Averaged across Layers)")
plt.xlabel("Key Position")
plt.ylabel("Query Position")
plt.tight_layout()
plt.show()

# Feature importance from feature-to-feature attention
importance = avg_map.sum(axis=0)  # total attention received by each key (feature)

n_features = len(feature_names)
feature_scores = [(i, importance[i]) for i in range(n_features)]
feature_scores.sort(key=lambda x: x[1], reverse=True)

top5 = feature_scores[:5]

print("\nTop 5 feature indices by feature-to-feature attention (most important first):")
for rank, (idx, score) in enumerate(top5, 1):
    name = feature_names[idx]
    print(f"{rank}: {name} (score={float(score):.6f})")

plt.figure(figsize=(10, 5))
labels = [feature_names[idx] for idx, _ in top5]
scores = [score for _, score in top5]
sns.barplot(x=labels, y=scores, hue=labels, legend=False, palette="viridis")
plt.title("Top 5 Feature Importances (Feature-to-Feature Attention)")
plt.ylabel("Importance (sum of attention received)")
plt.xlabel("Feature")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 2: Attention to Label (Label-to-Feature Attention)
# ==============================================================================
print("\n=== Attention to Label (Label-to-Feature Attention) ===")
label_attention = clf.get_attention_to_label()
print(f"Label Attention Shape: {label_attention.shape}")

# Create feature importance ranking from label attention
label_feature_scores = [(i, label_attention[i]) for i in range(n_features)]
label_feature_scores.sort(key=lambda x: x[1], reverse=True)

top5_label = label_feature_scores[:5]

print("\nTop 5 features by label attention (most important for prediction):")
for rank, (idx, score) in enumerate(top5_label, 1):
    name = feature_names[idx]
    print(f"{rank}: {name} (score={float(score):.6f})")

plt.figure(figsize=(10, 5))
labels_plot = [feature_names[idx] for idx, _ in top5_label]
scores_plot = [score for _, score in top5_label]
sns.barplot(x=labels_plot, y=scores_plot, hue=labels_plot, legend=False, palette="magma")
plt.title("Top 5 Feature Importances (Label Attention)")
plt.ylabel("Attention from Label to Feature")
plt.xlabel("Feature")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

# ==============================================================================
# PART 3: Comparison Plot
# ==============================================================================
print("\n=== Comparison: Feature-to-Feature vs Label Attention ===")

# Normalize both importance measures for comparison
feat_importance_norm = importance / importance.max()
label_importance_norm = label_attention / label_attention.max()

# Create comparison dataframe-like structure
comparison_data = []
for i in range(n_features):
    comparison_data.append(
        {
            "feature": feature_names[i],
            "feature_attention": feat_importance_norm[i],
            "label_attention": label_importance_norm[i],
        }
    )

# Sort by label attention
comparison_data.sort(key=lambda x: x["label_attention"], reverse=True)

# Show top 10 features
top10_comparison = comparison_data[:10]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(top10_comparison))
width = 0.35

bars1 = ax.bar(
    x - width / 2,
    [d["feature_attention"] for d in top10_comparison],
    width,
    label="Feature-to-Feature",
    color="steelblue",
)
bars2 = ax.bar(
    x + width / 2,
    [d["label_attention"] for d in top10_comparison],
    width,
    label="Label Attention",
    color="coral",
)

ax.set_ylabel("Normalized Importance")
ax.set_title("Top 10 Features: Feature-to-Feature vs Label Attention")
ax.set_xticks(x)
ax.set_xticklabels([d["feature"] for d in top10_comparison], rotation=30, ha="right")
ax.legend()
plt.tight_layout()
plt.show()

print("\nDone! Both attention types have been visualized.")
