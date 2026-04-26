"""
IDS Pipeline for NF-ToN-IoT-v2 Dataset
Steps:
  1. Data Cleaning & Preprocessing
  2. Modified Binary Grey Wolf Optimization (MBGWO) for Feature Selection
  3. ML Classification (XGBoost / Gradient Boosting)
  4. Comparison with Random Forest
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import time, random, math

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.utils import resample

# ─── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — DATA CLEANING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("STEP 1: Data Cleaning & Preprocessing")
print("=" * 65)

df = pd.read_csv("subset_05_NF-ToN-IoT-v2.csv")
print(f"  Raw shape          : {df.shape}")

# 1a. Drop IP address columns (not numeric, leak identifiers)
ip_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR"]
df.drop(columns=ip_cols, inplace=True)
print(f"  Dropped IP cols    : {ip_cols}")

# 1b. Drop 'Attack' column (multiclass label — we use binary 'Label')
df.drop(columns=["Attack"], inplace=True)

# 1c. Missing values
missing = df.isnull().sum().sum()
print(f"  Missing values     : {missing}")
if missing:
    df.fillna(df.median(numeric_only=True), inplace=True)

# 1d. Duplicate rows
dups = df.duplicated().sum()
print(f"  Duplicate rows     : {dups}")
df.drop_duplicates(inplace=True)

# 1e. Constant / near-zero-variance features
feature_cols = [c for c in df.columns if c != "Label"]
var = df[feature_cols].var()
low_var = var[var < 1e-6].index.tolist()
print(f"  Near-zero variance : {low_var}")
df.drop(columns=low_var, inplace=True)
feature_cols = [c for c in df.columns if c != "Label"]

# 1f. Stratified sample (keep computation manageable, preserve class ratio)
SAMPLE_SIZE = 80_000
df_sample, _ = train_test_split(df, train_size=SAMPLE_SIZE,
                                stratify=df["Label"], random_state=SEED)
print(f"  Stratified sample  : {df_sample.shape}")
print(f"  Class distribution : {df_sample['Label'].value_counts().to_dict()}")

X = df_sample[feature_cols].values.astype(np.float64)
y = df_sample["Label"].values

# 1g. Normalise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Features after cleaning: {X_scaled.shape[1]}")
print(f"  Normalization      : StandardScaler (zero-mean, unit-variance)")

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"  Train size         : {X_train.shape[0]}")
print(f"  Test  size         : {X_test.shape[0]}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — MODIFIED BINARY GREY WOLF OPTIMIZER (MBGWO)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 2: Modified Binary Grey Wolf Optimization (MBGWO)")
print("=" * 65)

class MBGWO:
    """
    Modified Binary Grey Wolf Optimizer for feature selection.

    Modifications over standard BGWO:
      1. V-shaped transfer function  (V-TF) instead of S-shaped for
         better exploration of binary search space.
      2. Lévy-flight position update for alpha wolf to escape local optima.
      3. Dynamic a-decay with sinusoidal perturbation for adaptive
         exploitation/exploration balance.
      4. Elitist preservation: best solution is always retained.
    """

    def __init__(self, n_wolves=20, max_iter=30, alpha_param=2.0,
                 fitness_alpha=0.99, fitness_beta=0.01, seed=42):
        self.n_wolves     = n_wolves
        self.max_iter     = max_iter
        self.alpha_param  = alpha_param   # initial a value
        self.fit_alpha    = fitness_alpha  # weight for classification accuracy
        self.fit_beta     = fitness_beta   # weight for feature reduction
        self.seed         = seed

    # ── transfer functions ────────────────────────────────────────────────────
    @staticmethod
    def _v_transfer(x):
        """V-shaped transfer: |tanh(x)| → maps real to [0,1]."""
        return np.abs(np.tanh(x))

    # ── Lévy flight ───────────────────────────────────────────────────────────
    @staticmethod
    def _levy(dim, beta=1.5):
        sigma = (math.gamma(1+beta)*math.sin(math.pi*beta/2) /
                 (math.gamma((1+beta)/2)*beta*2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(dim) * sigma
        v = np.abs(np.random.randn(dim))
        return u / (v ** (1/beta))

    # ── fitness function ──────────────────────────────────────────────────────
    def _fitness(self, mask, X_tr, y_tr, X_val, y_val):
        if mask.sum() == 0:
            return 1.0          # penalise empty selection
        clf = RandomForestClassifier(
            n_estimators=30, max_depth=6, n_jobs=-1, random_state=self.seed)
        clf.fit(X_tr[:, mask], y_tr)
        preds = clf.predict(X_val[:, mask])
        err   = 1.0 - accuracy_score(y_val, preds)
        feat_ratio = mask.sum() / len(mask)
        return self.fit_alpha * err + self.fit_beta * feat_ratio

    # ── main optimisation loop ────────────────────────────────────────────────
    def optimize(self, X_tr, y_tr, X_val, y_val):
        np.random.seed(self.seed)
        n_feat = X_tr.shape[1]

        # Initialise wolves with random binary positions
        pos = np.random.randint(0, 2, (self.n_wolves, n_feat)).astype(float)

        # Evaluate initial fitness
        fitness = np.array([
            self._fitness(pos[i].astype(bool), X_tr, y_tr, X_val, y_val)
            for i in range(self.n_wolves)])

        # Assign alpha, beta, delta
        order = np.argsort(fitness)
        alpha_pos, alpha_fit = pos[order[0]].copy(), fitness[order[0]]
        beta_pos,  beta_fit  = pos[order[1]].copy(), fitness[order[1]]
        delta_pos, delta_fit = pos[order[2]].copy(), fitness[order[2]]

        history = []
        print(f"  {'Iter':>4}  {'BestFit':>10}  {'Features':>10}")
        print(f"  {'-'*4}  {'-'*10}  {'-'*10}")

        for t in range(self.max_iter):
            # Modified a-decay: linear + sinusoidal perturbation
            a = self.alpha_param * (1 - t / self.max_iter) * (
                1 + 0.1 * math.sin(math.pi * t / self.max_iter))

            for i in range(self.n_wolves):
                r1, r2 = np.random.rand(n_feat), np.random.rand(n_feat)
                A1 = 2*a*r1 - a;  C1 = 2*r2
                r1, r2 = np.random.rand(n_feat), np.random.rand(n_feat)
                A2 = 2*a*r1 - a;  C2 = 2*r2
                r1, r2 = np.random.rand(n_feat), np.random.rand(n_feat)
                A3 = 2*a*r1 - a;  C3 = 2*r2

                D_alpha = np.abs(C1*alpha_pos - pos[i])
                D_beta  = np.abs(C2*beta_pos  - pos[i])
                D_delta = np.abs(C3*delta_pos - pos[i])

                X1 = alpha_pos - A1*D_alpha
                X2 = beta_pos  - A2*D_beta
                X3 = delta_pos - A3*D_delta

                new_pos_cont = (X1 + X2 + X3) / 3.0

                # V-shaped transfer to binary
                prob = self._v_transfer(new_pos_cont)
                pos[i] = (np.random.rand(n_feat) < prob).astype(float)

            # Lévy flight for alpha wolf (exploration boost)
            levy_step = self._levy(n_feat)
            alpha_new_cont = alpha_pos + levy_step * 0.01
            prob_alpha = self._v_transfer(alpha_new_cont)
            alpha_candidate = (np.random.rand(n_feat) < prob_alpha).astype(float)
            f_cand = self._fitness(alpha_candidate.astype(bool),
                                   X_tr, y_tr, X_val, y_val)
            if f_cand < alpha_fit:
                alpha_pos, alpha_fit = alpha_candidate.copy(), f_cand

            # Re-evaluate all wolves
            fitness = np.array([
                self._fitness(pos[i].astype(bool), X_tr, y_tr, X_val, y_val)
                for i in range(self.n_wolves)])

            # Update hierarchy
            order = np.argsort(fitness)
            if fitness[order[0]] < alpha_fit:
                alpha_pos, alpha_fit = pos[order[0]].copy(), fitness[order[0]]
            if fitness[order[1]] < beta_fit:
                beta_pos, beta_fit = pos[order[1]].copy(), fitness[order[1]]
            if fitness[order[2]] < delta_fit:
                delta_pos, delta_fit = pos[order[2]].copy(), fitness[order[2]]

            n_sel = int(alpha_pos.sum())
            history.append((t+1, alpha_fit, n_sel))
            if (t+1) % 5 == 0 or t == 0:
                print(f"  {t+1:>4}  {alpha_fit:>10.6f}  {n_sel:>10}")

        self.best_mask_    = alpha_pos.astype(bool)
        self.best_fitness_ = alpha_fit
        self.history_      = history
        selected = [feature_cols[i] for i in range(n_feat)
                    if self.best_mask_[i]]
        print(f"\n  Selected features  : {len(selected)} / {n_feat}")
        print(f"  Best fitness score : {self.best_fitness_:.6f}")
        print(f"  Selected           : {selected}")
        return self.best_mask_


# Split training set further for MBGWO fitness evaluation
X_tr_gwo, X_val_gwo, y_tr_gwo, y_val_gwo = train_test_split(
    X_train, y_train, test_size=0.25, stratify=y_train, random_state=SEED)

mbgwo = MBGWO(n_wolves=20, max_iter=30, alpha_param=2.0,
              fitness_alpha=0.99, fitness_beta=0.01, seed=SEED)

t0 = time.time()
best_mask = mbgwo.optimize(X_tr_gwo, y_tr_gwo, X_val_gwo, y_val_gwo)
gwo_time = time.time() - t0
print(f"  MBGWO runtime      : {gwo_time:.1f}s")

# Apply mask to train/test
X_train_sel = X_train[:, best_mask]
X_test_sel  = X_test[:, best_mask]
selected_feature_names = [feature_cols[i] for i in range(len(feature_cols))
                           if best_mask[i]]

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — ML CLASSIFICATION (Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 3: ML Classification — Gradient Boosting Classifier")
print("=" * 65)

gb_clf = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.1, max_depth=5,
    subsample=0.8, random_state=SEED)

t0 = time.time()
gb_clf.fit(X_train_sel, y_train)
gb_train_time = time.time() - t0

y_pred_gb = gb_clf.predict(X_test_sel)

gb_acc  = accuracy_score(y_test, y_pred_gb)
gb_prec = precision_score(y_test, y_pred_gb, zero_division=0)
gb_rec  = recall_score(y_test, y_pred_gb, zero_division=0)
gb_f1   = f1_score(y_test, y_pred_gb, zero_division=0)
gb_cm   = confusion_matrix(y_test, y_pred_gb)

print(f"  Accuracy  : {gb_acc:.4f}")
print(f"  Precision : {gb_prec:.4f}")
print(f"  Recall    : {gb_rec:.4f}")
print(f"  F1-Score  : {gb_f1:.4f}")
print(f"  Train time: {gb_train_time:.1f}s")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_gb,
      target_names=["Benign", "Attack"]))

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — RANDOM FOREST (same selected features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("STEP 4: Comparison — Random Forest Classifier")
print("=" * 65)

rf_clf = RandomForestClassifier(
    n_estimators=150, max_depth=None, n_jobs=-1, random_state=SEED)

t0 = time.time()
rf_clf.fit(X_train_sel, y_train)
rf_train_time = time.time() - t0

y_pred_rf = rf_clf.predict(X_test_sel)

rf_acc  = accuracy_score(y_test, y_pred_rf)
rf_prec = precision_score(y_test, y_pred_rf, zero_division=0)
rf_rec  = recall_score(y_test, y_pred_rf, zero_division=0)
rf_f1   = f1_score(y_test, y_pred_rf, zero_division=0)
rf_cm   = confusion_matrix(y_test, y_pred_rf)

print(f"  Accuracy  : {rf_acc:.4f}")
print(f"  Precision : {rf_prec:.4f}")
print(f"  Recall    : {rf_rec:.4f}")
print(f"  F1-Score  : {rf_f1:.4f}")
print(f"  Train time: {rf_train_time:.1f}s")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_rf,
      target_names=["Benign", "Attack"]))

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("COMPARISON SUMMARY")
print("=" * 65)
summary = pd.DataFrame({
    "Model":     ["MBGWO + Gradient Boosting", "MBGWO + Random Forest"],
    "Accuracy":  [gb_acc,  rf_acc],
    "Precision": [gb_prec, rf_prec],
    "Recall":    [gb_rec,  rf_rec],
    "F1-Score":  [gb_f1,   rf_f1],
    "Train(s)":  [gb_train_time, rf_train_time],
})
print(summary.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════
sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure(figsize=(20, 16))
fig.suptitle("IDS Pipeline — NF-ToN-IoT-v2 Results", fontsize=16,
             fontweight="bold", y=0.98)

# ── (a) MBGWO Convergence ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(3, 3, 1)
iters, fits, feats = zip(*mbgwo.history_)
ax1.plot(iters, fits, "o-", color="#1f77b4", linewidth=2, markersize=4)
ax1.set_title("MBGWO Convergence", fontweight="bold")
ax1.set_xlabel("Iteration"); ax1.set_ylabel("Best Fitness")
ax1.fill_between(iters, fits, alpha=0.15, color="#1f77b4")

# ── (b) Feature count over iterations ────────────────────────────────────────
ax2 = fig.add_subplot(3, 3, 2)
ax2.plot(iters, feats, "s-", color="#ff7f0e", linewidth=2, markersize=4)
ax2.set_title("Selected Features per Iteration", fontweight="bold")
ax2.set_xlabel("Iteration"); ax2.set_ylabel("# Features Selected")
ax2.axhline(len(selected_feature_names), ls="--", color="gray", label=f"Final={len(selected_feature_names)}")
ax2.legend()

# ── (c) Bar chart — metrics comparison ───────────────────────────────────────
ax3 = fig.add_subplot(3, 3, 3)
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
gb_vals = [gb_acc, gb_prec, gb_rec, gb_f1]
rf_vals = [rf_acc, rf_prec, rf_rec, rf_f1]
x = np.arange(len(metrics)); w = 0.35
ax3.bar(x - w/2, gb_vals, w, label="Gradient Boosting", color="#1f77b4", edgecolor="white")
ax3.bar(x + w/2, rf_vals, w, label="Random Forest",     color="#ff7f0e", edgecolor="white")
ax3.set_xticks(x); ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_ylim(0.85, 1.01)
ax3.set_title("Model Performance Comparison", fontweight="bold")
ax3.set_ylabel("Score"); ax3.legend(fontsize=8)
for xi, (g, r) in enumerate(zip(gb_vals, rf_vals)):
    ax3.text(xi-w/2, g+0.002, f"{g:.3f}", ha="center", va="bottom", fontsize=7)
    ax3.text(xi+w/2, r+0.002, f"{r:.3f}", ha="center", va="bottom", fontsize=7)

# ── (d) Confusion matrix — GB ─────────────────────────────────────────────────
ax4 = fig.add_subplot(3, 3, 4)
sns.heatmap(gb_cm, annot=True, fmt="d", cmap="Blues", ax=ax4,
            xticklabels=["Benign","Attack"], yticklabels=["Benign","Attack"])
ax4.set_title("Confusion Matrix — Gradient Boosting", fontweight="bold")
ax4.set_xlabel("Predicted"); ax4.set_ylabel("Actual")

# ── (e) Confusion matrix — RF ─────────────────────────────────────────────────
ax5 = fig.add_subplot(3, 3, 5)
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Oranges", ax=ax5,
            xticklabels=["Benign","Attack"], yticklabels=["Benign","Attack"])
ax5.set_title("Confusion Matrix — Random Forest", fontweight="bold")
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")

# ── (f) Feature importance (RF) for selected features ────────────────────────
ax6 = fig.add_subplot(3, 3, 6)
fi = pd.Series(rf_clf.feature_importances_, index=selected_feature_names)
fi_top = fi.nlargest(15)
fi_top.sort_values().plot(kind="barh", ax=ax6, color="#2ca02c", edgecolor="white")
ax6.set_title("Top-15 Feature Importances (RF)", fontweight="bold")
ax6.set_xlabel("Importance")

# ── (g) Class distribution ────────────────────────────────────────────────────
ax7 = fig.add_subplot(3, 3, 7)
vals   = [int((y==0).sum()), int((y==1).sum())]
labels = ["Benign", "Attack"]
colors = ["#aec7e8", "#ffbb78"]
wedges, texts, autotexts = ax7.pie(vals, labels=labels, autopct="%1.1f%%",
                                    colors=colors, startangle=90,
                                    wedgeprops={"edgecolor":"white","linewidth":1.5})
ax7.set_title("Class Distribution (Sample)", fontweight="bold")

# ── (h) Selected vs dropped features ─────────────────────────────────────────
ax8 = fig.add_subplot(3, 3, 8)
n_sel = int(best_mask.sum())
n_tot = len(best_mask)
n_drop = n_tot - n_sel
ax8.bar(["Selected", "Dropped"], [n_sel, n_drop],
        color=["#1f77b4", "#d62728"], edgecolor="white", width=0.5)
ax8.set_title(f"Feature Reduction\n({n_sel}/{n_tot} selected)", fontweight="bold")
ax8.set_ylabel("Number of Features")
for i, v in enumerate([n_sel, n_drop]):
    ax8.text(i, v+0.2, str(v), ha="center", fontweight="bold")

# ── (i) Radar / spider chart — metrics ───────────────────────────────────────
ax9 = fig.add_subplot(3, 3, 9, polar=True)
categories = ["Accuracy", "Precision", "Recall", "F1"]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
gb_plot = gb_vals + gb_vals[:1]
rf_plot = rf_vals + rf_vals[:1]
ax9.set_xticks(angles[:-1])
ax9.set_xticklabels(categories, fontsize=9)
ax9.set_ylim(0.85, 1.0)
ax9.plot(angles, gb_plot, "o-", linewidth=2, label="GB", color="#1f77b4")
ax9.fill(angles, gb_plot, alpha=0.1, color="#1f77b4")
ax9.plot(angles, rf_plot, "s-", linewidth=2, label="RF", color="#ff7f0e")
ax9.fill(angles, rf_plot, alpha=0.1, color="#ff7f0e")
ax9.set_title("Radar — Performance", fontweight="bold", pad=15)
ax9.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("ids_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Plot saved → ids_results.png")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE RESULTS CSV
# ══════════════════════════════════════════════════════════════════════════════
summary.to_csv("model_comparison.csv", index=False)

feat_df = pd.DataFrame({
    "Feature": selected_feature_names,
    "RF_Importance": rf_clf.feature_importances_
}).sort_values("RF_Importance", ascending=False)
feat_df.to_csv("selected_features.csv", index=False)


print("\n  Results CSV saved → model_comparison.csv")
print("  Features CSV saved → selected_features.csv")
print("\nDone ✓")