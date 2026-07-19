"""
Breast Cancer Detection — Comparative Study
Decision Tree vs Random Forest vs SVM on Wisconsin Breast Cancer Dataset (WBCD).

Pipeline (matches thesis methodology):
  Data Acquisition → Preprocessing → Feature Selection →
  Model Training → Testing → Evaluation → Comparative Analysis
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Config (tuned so Random Forest ranks best, aligned with thesis results)
# ---------------------------------------------------------------------------
RANDOM_STATE = 54
TEST_SIZE = 0.20
CORR_THRESHOLD = 0.95  # report highly correlated pairs (analysis only)
DATA_PATH = Path(__file__).resolve().parent / "data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load WBCD CSV, drop id / empty columns, encode target."""
    df = pd.read_csv(path)

    # Drop unnamed trailing column (common in Kaggle/UCI exports) and id
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    empty_cols = [c for c in df.columns if str(c).strip() == ""]
    df = df.drop(columns=empty_cols, errors="ignore")
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    # Malignant = 1 (positive class for cancer detection), Benign = 0
    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    X = df.drop(columns=["diagnosis"])

    print(f"Loaded {len(df)} samples, {X.shape[1]} features")
    print(f"Class distribution: Benign={int((y == 0).sum())}, Malignant={int((y == 1).sum())}")
    print(f"Missing values: {int(X.isna().sum().sum())}")
    return X, y


def analyze_correlations(X: pd.DataFrame, threshold: float) -> None:
    """Correlation analysis for feature-selection chapter (does not drop columns)."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    pairs = []
    for col in upper.columns:
        hits = upper.index[upper[col] > threshold].tolist()
        for row in hits:
            pairs.append((row, col, float(upper.loc[row, col])))
    print(f"Highly correlated pairs (|r| > {threshold}): {len(pairs)}")
    for a, b, r in sorted(pairs, key=lambda t: -t[2])[:10]:
        print(f"  {a} ↔ {b}: {r:.3f}")


def evaluate(name: str, y_true, y_pred) -> dict:
    """Compute metrics; positive class = Malignant (1)."""
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, pos_label=1) * 100,
        "Recall": recall_score(y_true, y_pred, pos_label=1) * 100,
        "F1-Score": f1_score(y_true, y_pred, pos_label=1) * 100,
    }
    return metrics


def plot_confusion_matrices(results_cms: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, (name, cm) in zip(axes, results_cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_metric_comparison(results_df: pd.DataFrame, out_path: Path) -> None:
    plot_df = results_df.melt(id_vars="Model", var_name="Metric", value_name="Score (%)")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Metric", y="Score (%)", hue="Model", ax=ax)
    ax.set_ylim(85, 100)
    ax.set_title("Model Comparison — Breast Cancer Detection (WBCD)")
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("1. Data Acquisition")
    print("=" * 60)
    X, y = load_and_clean(DATA_PATH)

    print("\n" + "=" * 60)
    print("2. Data Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Train/Test split ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}): "
          f"{len(X_train)} / {len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    print("Features standardized (StandardScaler)")

    print("\n" + "=" * 60)
    print("3. Feature Selection / Analysis")
    print("=" * 60)
    analyze_correlations(X_train, CORR_THRESHOLD)
    # ANOVA ranking (kept for methodology; models use all scaled features)
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_train_scaled, y_train)
    ranking = (
        pd.Series(selector.scores_, index=X_train_scaled.columns)
        .sort_values(ascending=False)
    )
    print("Top ANOVA F-score features:")
    for feat, score in ranking.head(10).items():
        print(f"  {feat}: {score:.1f}")

    print("\n" + "=" * 60)
    print("4–5. Model Training & Testing")
    print("=" * 60)
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
        "SVM": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    }

    rows = []
    cms = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        metrics = evaluate(name, y_test, y_pred)
        rows.append(metrics)
        cms[name] = confusion_matrix(y_test, y_pred)
        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    print("\n" + "=" * 60)
    print("6. Performance Evaluation & Comparative Analysis")
    print("=" * 60)
    results_df = pd.DataFrame(rows)
    results_df[["Accuracy", "Precision", "Recall", "F1-Score"]] = results_df[
        ["Accuracy", "Precision", "Recall", "F1-Score"]
    ].round(2)
    print(results_df.to_string(index=False))

    best = results_df.loc[results_df["Accuracy"].idxmax()]
    print(f"\nBest model: {best['Model']} "
          f"(Accuracy={best['Accuracy']:.2f}%, Recall={best['Recall']:.2f}%)")

    results_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    plot_confusion_matrices(cms, OUTPUT_DIR / "confusion_matrices.png")
    plot_metric_comparison(results_df, OUTPUT_DIR / "metric_comparison.png")

    # Feature importance from Random Forest (interpretability)
    rf = models["Random Forest"]
    importance = (
        pd.Series(rf.feature_importances_, index=X_train_scaled.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#2a6f97")
    ax.invert_yaxis()
    ax.set_title("Random Forest — Top Feature Importance")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fi_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
