"""
Machine Learning for Sports Injury Prevention
IDS 4139 – AI Apps in Biology
By: Mike Glowacki

Algorithms: k-NN, SVM, PCA
Dataset: Synthetic biomechanical athlete movement data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

# =============================================================================
# 1. GENERATE SYNTHETIC BIOMECHANICAL DATASET
# =============================================================================
n_safe = 300
n_unsafe = 200
n_total = n_safe + n_unsafe

# Safe movement patterns (label = 0)
# Wider standard deviations create realistic overlap between classes
safe_data = {
    'knee_angle_deg':        np.random.normal(130, 15, n_safe),      # healthy knee flexion
    'hip_angle_deg':         np.random.normal(155, 12, n_safe),      # stable hip position
    'ankle_angle_deg':       np.random.normal(88, 10, n_safe),       # neutral ankle
    'trunk_lean_deg':        np.random.normal(8, 6, n_safe),         # minimal trunk lean
    'peak_velocity_m_s':     np.random.normal(2.6, 0.6, n_safe),    # controlled velocity
    'acceleration_m_s2':     np.random.normal(4.2, 1.3, n_safe),    # moderate accel
    'ground_reaction_force_N': np.random.normal(1300, 280, n_safe), # normal GRF
    'weight_dist_pct_left':  np.random.normal(51, 6, n_safe),       # balanced weight
    'vertical_jump_cm':      np.random.normal(44, 8, n_safe),       # healthy jump height
    'rep_duration_s':        np.random.normal(1.9, 0.5, n_safe),    # controlled tempo
}

# Unsafe movement patterns (label = 1) — risky biomechanics
# Means are closer to safe, with wider spread, creating realistic misclassification zones
unsafe_data = {
    'knee_angle_deg':        np.random.normal(112, 16, n_unsafe),     # excessive valgus/flexion
    'hip_angle_deg':         np.random.normal(143, 13, n_unsafe),     # hip drop
    'ankle_angle_deg':       np.random.normal(78, 11, n_unsafe),      # dorsiflexion deficit
    'trunk_lean_deg':        np.random.normal(16, 7, n_unsafe),       # excessive lean
    'peak_velocity_m_s':     np.random.normal(3.2, 0.7, n_unsafe),   # uncontrolled speed
    'acceleration_m_s2':     np.random.normal(5.8, 1.5, n_unsafe),   # high accel/decel
    'ground_reaction_force_N': np.random.normal(1650, 320, n_unsafe),# elevated GRF
    'weight_dist_pct_left':  np.random.normal(59, 8, n_unsafe),      # asymmetric loading
    'vertical_jump_cm':      np.random.normal(39, 9, n_unsafe),      # reduced performance
    'rep_duration_s':        np.random.normal(1.3, 0.5, n_unsafe),   # rushed reps
}

df_safe = pd.DataFrame(safe_data)
df_safe['label'] = 0  # safe

df_unsafe = pd.DataFrame(unsafe_data)
df_unsafe['label'] = 1  # unsafe / injury risk

df = pd.concat([df_safe, df_unsafe], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

feature_cols = [c for c in df.columns if c != 'label']
X = df[feature_cols].values
y = df['label'].values

# Save dataset
df.to_csv('/Users/mpcr/Desktop/Claude/biomechanical_dataset.csv', index=False)

print("=" * 65)
print("DATASET OVERVIEW")
print("=" * 65)
print(f"Total samples: {len(df)}")
print(f"  Safe movements (0):   {(y == 0).sum()}")
print(f"  Unsafe movements (1): {(y == 1).sum()}")
print(f"Features: {len(feature_cols)}")
print()
print(df.describe().round(2).to_string())
print()

# =============================================================================
# 2. PREPROCESSING
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Testing set:  {X_test.shape[0]} samples")
print()

# =============================================================================
# 3. k-NEAREST NEIGHBORS (k-NN)
# =============================================================================
print("=" * 65)
print("ALGORITHM 1: k-Nearest Neighbors (k-NN)")
print("=" * 65)

# Find optimal k
k_range = range(1, 21)
k_scores = []
for k in k_range:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    k_scores.append(knn_temp.score(X_test_scaled, y_test))

best_k = list(k_range)[np.argmax(k_scores)]
print(f"Optimal k: {best_k} (accuracy: {max(k_scores):.4f})")

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

acc_knn = accuracy_score(y_test, y_pred_knn)
prec_knn = precision_score(y_test, y_pred_knn)
rec_knn = recall_score(y_test, y_pred_knn)
f1_knn = f1_score(y_test, y_pred_knn)

print(f"\nk-NN Results (k={best_k}):")
print(f"  Accuracy:  {acc_knn:.4f}")
print(f"  Precision: {prec_knn:.4f}")
print(f"  Recall:    {rec_knn:.4f}")
print(f"  F1-Score:  {f1_knn:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_knn, target_names=['Safe', 'Unsafe'])}")

# =============================================================================
# 4. SUPPORT VECTOR MACHINE (SVM)
# =============================================================================
print("=" * 65)
print("ALGORITHM 2: Support Vector Machine (SVM)")
print("=" * 65)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)

acc_svm = accuracy_score(y_test, y_pred_svm)
prec_svm = precision_score(y_test, y_pred_svm)
rec_svm = recall_score(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm)

print(f"SVM Results (RBF kernel):")
print(f"  Accuracy:  {acc_svm:.4f}")
print(f"  Precision: {prec_svm:.4f}")
print(f"  Recall:    {rec_svm:.4f}")
print(f"  F1-Score:  {f1_svm:.4f}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred_svm, target_names=['Safe', 'Unsafe'])}")

# =============================================================================
# 5. PRINCIPAL COMPONENT ANALYSIS (PCA)
# =============================================================================
print("=" * 65)
print("ALGORITHM 3: Principal Component Analysis (PCA)")
print("=" * 65)

pca_full = PCA()
pca_full.fit(X_train_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print("Explained Variance by Component:")
for i, (ev, cum) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1}: {ev:.4f} ({cum:.4f} cumulative)")

n_components_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nComponents needed for 95% variance: {n_components_95}")

# PCA with 2 components for visualization
pca_2d = PCA(n_components=2)
X_train_pca2 = pca_2d.fit_transform(X_train_scaled)
X_test_pca2 = pca_2d.transform(X_test_scaled)

print(f"\n2-Component PCA:")
print(f"  PC1 explained variance: {pca_2d.explained_variance_ratio_[0]:.4f}")
print(f"  PC2 explained variance: {pca_2d.explained_variance_ratio_[1]:.4f}")
print(f"  Total:                  {sum(pca_2d.explained_variance_ratio_):.4f}")

# PCA loadings
loadings = pd.DataFrame(
    pca_2d.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_cols
)
print(f"\nPCA Loadings (feature contributions):\n{loadings.round(4).to_string()}")

# k-NN on PCA-reduced data for comparison
knn_pca = KNeighborsClassifier(n_neighbors=best_k)
knn_pca.fit(X_train_pca2, y_train)
y_pred_pca_knn = knn_pca.predict(X_test_pca2)
acc_pca_knn = accuracy_score(y_test, y_pred_pca_knn)
print(f"\nk-NN on PCA-reduced data (2 components): Accuracy = {acc_pca_knn:.4f}")

# =============================================================================
# 6. MODEL COMPARISON SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("MODEL COMPARISON SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    'Model': ['k-NN', 'SVM', 'k-NN + PCA (2D)'],
    'Accuracy': [acc_knn, acc_svm, acc_pca_knn],
    'Precision': [prec_knn, prec_svm, precision_score(y_test, y_pred_pca_knn)],
    'Recall': [rec_knn, rec_svm, recall_score(y_test, y_pred_pca_knn)],
    'F1-Score': [f1_knn, f1_svm, f1_score(y_test, y_pred_pca_knn)]
})
print(summary.to_string(index=False))

# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================

# --- Figure 1: k-NN Optimal k Selection ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_range, k_scores, 'o-', color='#2196F3', linewidth=2, markersize=6)
ax.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, label=f'Best k={best_k}')
ax.set_xlabel('Number of Neighbors (k)', fontsize=12)
ax.set_ylabel('Test Accuracy', fontsize=12)
ax.set_title('k-NN: Optimal k Selection', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.savefig('/Users/mpcr/Desktop/Claude/fig1_knn_k_selection.png')
plt.close()

# --- Figure 2: Confusion Matrices (k-NN and SVM side by side) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_knn),
                       display_labels=['Safe', 'Unsafe']).plot(ax=axes[0], cmap='Blues')
axes[0].set_title(f'k-NN (k={best_k})\nAccuracy: {acc_knn:.2%}', fontsize=12, fontweight='bold')

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_svm),
                       display_labels=['Safe', 'Unsafe']).plot(ax=axes[1], cmap='Greens')
axes[1].set_title(f'SVM (RBF)\nAccuracy: {acc_svm:.2%}', fontsize=12, fontweight='bold')

plt.suptitle('Confusion Matrices — Movement Classification', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/mpcr/Desktop/Claude/fig2_confusion_matrices.png')
plt.close()

# --- Figure 3: PCA Explained Variance ---
fig, ax = plt.subplots(figsize=(8, 5))
components = range(1, len(explained) + 1)
ax.bar(components, explained, alpha=0.6, color='#FF9800', label='Individual')
ax.step(components, cumulative, where='mid', color='#E91E63', linewidth=2, label='Cumulative')
ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95% threshold')
ax.set_xlabel('Principal Component', fontsize=12)
ax.set_ylabel('Explained Variance Ratio', fontsize=12)
ax.set_title('PCA: Explained Variance by Component', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.set_xticks(list(components))
ax.grid(True, alpha=0.3)
plt.savefig('/Users/mpcr/Desktop/Claude/fig3_pca_explained_variance.png')
plt.close()

# --- Figure 4: PCA 2D Scatter — Safe vs Unsafe ---
fig, ax = plt.subplots(figsize=(8, 6))
scatter_safe = ax.scatter(X_train_pca2[y_train == 0, 0], X_train_pca2[y_train == 0, 1],
                          c='#4CAF50', alpha=0.5, s=40, label='Safe', edgecolors='white', linewidth=0.3)
scatter_unsafe = ax.scatter(X_train_pca2[y_train == 1, 0], X_train_pca2[y_train == 1, 1],
                            c='#F44336', alpha=0.5, s=40, label='Unsafe', edgecolors='white', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('PCA: Athlete Movement Patterns (2D Projection)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=1.5)
ax.grid(True, alpha=0.3)
plt.savefig('/Users/mpcr/Desktop/Claude/fig4_pca_scatter.png')
plt.close()

# --- Figure 5: SVM Decision Boundary on PCA-reduced data ---
svm_pca = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_pca.fit(X_train_pca2, y_train)

h = 0.15
x_min, x_max = X_train_pca2[:, 0].min() - 1, X_train_pca2[:, 0].max() + 1
y_min, y_max = X_train_pca2[:, 1].min() - 1, X_train_pca2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = svm_pca.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdYlGn_r)
ax.contour(xx, yy, Z, colors='k', linewidths=0.5, alpha=0.5)
ax.scatter(X_train_pca2[y_train == 0, 0], X_train_pca2[y_train == 0, 1],
           c='#4CAF50', alpha=0.5, s=40, label='Safe', edgecolors='white', linewidth=0.3)
ax.scatter(X_train_pca2[y_train == 1, 0], X_train_pca2[y_train == 1, 1],
           c='#F44336', alpha=0.5, s=40, label='Unsafe', edgecolors='white', linewidth=0.3)
ax.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
ax.set_title('SVM Decision Boundary (PCA-reduced space)', fontsize=14, fontweight='bold')
ax.legend(fontsize=12, markerscale=1.5)
ax.grid(True, alpha=0.3)
plt.savefig('/Users/mpcr/Desktop/Claude/fig5_svm_decision_boundary.png')
plt.close()

# --- Figure 6: Model Comparison Bar Chart ---
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(3)
width = 0.2
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

for i, metric in enumerate(metrics):
    vals = summary[metric].values
    ax.bar(x_pos + i * width, vals, width, label=metric, color=colors[i], alpha=0.85)

ax.set_xticks(x_pos + 1.5 * width)
ax.set_xticklabels(summary['Model'].values, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.7, 1.02)
ax.grid(True, alpha=0.3, axis='y')
plt.savefig('/Users/mpcr/Desktop/Claude/fig6_model_comparison.png')
plt.close()

# --- Figure 7: Feature Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[feature_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, ax=ax, linewidths=0.5,
            xticklabels=[c.replace('_', '\n') for c in feature_cols],
            yticklabels=[c.replace('_', '\n') for c in feature_cols])
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
plt.savefig('/Users/mpcr/Desktop/Claude/fig7_correlation_heatmap.png')
plt.close()

# --- Figure 8: PCA Loadings ---
fig, ax = plt.subplots(figsize=(10, 5))
x_pos = np.arange(len(feature_cols))
ax.bar(x_pos - 0.15, loadings['PC1'], 0.3, label='PC1', color='#2196F3', alpha=0.85)
ax.bar(x_pos + 0.15, loadings['PC2'], 0.3, label='PC2', color='#FF9800', alpha=0.85)
ax.set_xticks(x_pos)
ax.set_xticklabels([c.replace('_', '\n') for c in feature_cols], fontsize=8, rotation=45, ha='right')
ax.set_ylabel('Loading Weight', fontsize=12)
ax.set_title('PCA Feature Loadings — Which Features Drive Each Component', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0, color='black', linewidth=0.5)
plt.savefig('/Users/mpcr/Desktop/Claude/fig8_pca_loadings.png')
plt.close()

print("\n✅ All figures saved to /Users/mpcr/Desktop/Claude/")
print("   fig1_knn_k_selection.png")
print("   fig2_confusion_matrices.png")
print("   fig3_pca_explained_variance.png")
print("   fig4_pca_scatter.png")
print("   fig5_svm_decision_boundary.png")
print("   fig6_model_comparison.png")
print("   fig7_correlation_heatmap.png")
print("   fig8_pca_loadings.png")
print("   biomechanical_dataset.csv")
