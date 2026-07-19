"""
Breast Cancer Detection — Comparative Study (Optimized)
Decision Tree vs Random Forest vs SVM on Wisconsin Breast Cancer Dataset (WBCD).

Optimized pipeline:
  Data Acquisition → Preprocessing → Correlation filter →
  Nested CV hyperparameter tuning (Pipeline) → Hold-out test → Comparison

Key improvements over the baseline:
  - sklearn Pipeline (scaler + SelectKBest + model) to prevent leakage
  - Drop redundant highly correlated features (fit on train only)
  - Stratified K-Fold + GridSearchCV; tune for malignant-class F1
  - class_weight='balanced' for mild class imbalance
  - Final metrics reported on a held-out test set
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
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5
CORR_THRESHOLD = 0.92
DATA_PATH = Path(__file__).resolve().parent / "data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Tune for malignant-class F1 (balance precision/recall for cancer detection)
F1_MALIGNANT = make_scorer(f1_score, pos_label=1)


def load_and_clean(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load WBCD CSV, drop id / empty columns, encode target."""
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    empty_cols = [c for c in df.columns if str(c).strip() == ""]
    df = df.drop(columns=empty_cols, errors="ignore")
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    y = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    X = df.drop(columns=["diagnosis"])

    print(f"Loaded {len(df)} samples, {X.shape[1]} features")
    print(
        f"Class distribution: Benign={int((y == 0).sum())}, "
        f"Malignant={int((y == 1).sum())}"
    )
    print(f"Missing values: {int(X.isna().sum().sum())}")
    return X, y


def drop_highly_correlated(
    X_train: pd.DataFrame, X_test: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Drop redundant features using correlations from the training set only."""
    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    kept_cols = [c for c in X_train.columns if c not in to_drop]
    print(
        f"Correlation filter (>{threshold}): dropped {len(to_drop)} → "
        f"{len(kept_cols)} features remain"
    )
    if to_drop:
        print(f"  Dropped: {to_drop}")
    return X_train[kept_cols].copy(), X_test[kept_cols].copy(), kept_cols


def evaluate(name: str, y_true, y_pred) -> dict:
    """Compute hold-out metrics; positive class = Malignant (1)."""
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred) * 100,
        "Precision": precision_score(y_true, y_pred, pos_label=1) * 100,
        "Recall": recall_score(y_true, y_pred, pos_label=1) * 100,
        "F1-Score": f1_score(y_true, y_pred, pos_label=1) * 100,
    }


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
    plot_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    plot_df = results_df[plot_cols].melt(
        id_vars="Model", var_name="Metric", value_name="Score (%)"
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Metric", y="Score (%)", hue="Model", ax=ax)
    ax.set_ylim(85, 100)
    ax.set_title("Optimized Model Comparison — Breast Cancer Detection (WBCD)")
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def build_search_spaces(n_features: int) -> dict[str, tuple[Pipeline, dict]]:
    """Pipelines + grids. SelectKBest k is tuned inside CV."""
    k_options = sorted({min(k, n_features) for k in (10, 15, 20, n_features)})

    dt = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE)),
        ]
    )
    dt_grid = {
        "select__k": k_options,
        "clf__max_depth": [3, 5, 8, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__class_weight": [None, "balanced"],
    }

    rf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            (
                "clf",
                RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf_grid = {
        "select__k": k_options,
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [5, 10, None],
        "clf__min_samples_leaf": [1, 2],
        "clf__max_features": ["sqrt", 0.5],
        "clf__class_weight": [None, "balanced"],
    }

    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif)),
            ("clf", SVC(kernel="rbf", random_state=RANDOM_STATE)),
        ]
    )
    svm_grid = {
        "select__k": k_options,
        "clf__C": [0.1, 1.0, 10.0],
        "clf__gamma": ["scale", 0.01, 0.1],
        "clf__class_weight": [None, "balanced"],
    }

    return {
        "Decision Tree": (dt, dt_grid),
        "Random Forest": (rf, rf_grid),
        "SVM": (svm, svm_grid),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")

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
    print(
        f"Train/Test split ({int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}): "
        f"{len(X_train)} / {len(X_test)}"
    )

    print("\n" + "=" * 60)
    print("3. Feature Selection (correlation + ANOVA inside CV)")
    print("=" * 60)
    X_train, X_test, feature_names = drop_highly_correlated(
        X_train, X_test, CORR_THRESHOLD
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    search_spaces = build_search_spaces(n_features=X_train.shape[1])

    print("\n" + "=" * 60)
    print("4–5. Nested CV Tuning + Hold-out Testing")
    print("=" * 60)
    print(f"CV: {N_SPLITS}-fold stratified | scoring: malignant F1")

    rows = []
    cms = {}
    fitted = {}

    for name, (pipe, grid) in search_spaces.items():
        print(f"\n--- Tuning {name} ({len(grid)} param groups) ---")
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=F1_MALIGNANT,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        fitted[name] = best

        y_pred = best.predict(X_test)
        metrics = evaluate(name, y_test, y_pred)
        metrics["CV_F1_mean"] = round(search.best_score_ * 100, 2)
        metrics["CV_F1_std"] = round(
            search.cv_results_["std_test_score"][search.best_index_] * 100, 2
        )
        rows.append(metrics)
        cms[name] = confusion_matrix(y_test, y_pred)

        print(f"Best params: {search.best_params_}")
        print(
            f"CV F1 (malignant): {metrics['CV_F1_mean']:.2f}% "
            f"± {metrics['CV_F1_std']:.2f}%"
        )
        print(classification_report(y_test, y_pred, target_names=["Benign", "Malignant"]))

    print("\n" + "=" * 60)
    print("6. Performance Evaluation & Comparative Analysis")
    print("=" * 60)
    results_df = pd.DataFrame(rows)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "CV_F1_mean", "CV_F1_std"]
    results_df[metric_cols] = results_df[metric_cols].round(2)
    print(results_df.to_string(index=False))

    best_row = results_df.loc[results_df["F1-Score"].idxmax()]
    print(
        f"\nBest model (hold-out F1): {best_row['Model']} "
        f"(Accuracy={best_row['Accuracy']:.2f}%, "
        f"Recall={best_row['Recall']:.2f}%, F1={best_row['F1-Score']:.2f}%)"
    )

    results_path = OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved: {results_path}")

    plot_confusion_matrices(cms, OUTPUT_DIR / "confusion_matrices.png")
    plot_metric_comparison(results_df, OUTPUT_DIR / "metric_comparison.png")

    # Feature importance from the tuned Random Forest (selected features)
    rf_pipe = fitted["Random Forest"]
    selector: SelectKBest = rf_pipe.named_steps["select"]
    rf_clf: RandomForestClassifier = rf_pipe.named_steps["clf"]
    selected = [feature_names[i] for i in selector.get_support(indices=True)]
    importance = (
        pd.Series(rf_clf.feature_importances_, index=selected)
        .sort_values(ascending=False)
        .head(15)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", ax=ax, color="#2a6f97")
    ax.invert_yaxis()
    ax.set_title("Random Forest — Top Feature Importance (tuned)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    fi_path = OUTPUT_DIR / "feature_importance.png"
    fig.savefig(fi_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fi_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
