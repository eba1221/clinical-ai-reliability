"""Week 2 baseline: logistic regression and random forest on the full development pool.

What this script does, in order:
  1. Loads the dataset and the fixed split made by src/split.py.
  2. Trains two models on ALL 398 development patients (the pool):
       - logistic regression (L2 penalty, standardised features)
       - random forest
     Each model's one tuning setting is chosen by 5-fold cross-validation INSIDE
     the development pool. Standardisation sits inside the pipeline, so it is fitted
     on each training fold only. The test set plays no part in training or tuning.
  3. Grades both models once on the fixed 171-patient test set:
       discrimination:  AUROC, accuracy, sensitivity, specificity (threshold 0.5)
       calibration:     Brier score, log loss, calibration intercept and slope,
                        expected calibration error (ECE)
  4. Adds 95% bootstrap intervals (resampling the test patients) for AUROC, Brier
     score and accuracy, so you can see how uncertain 171 patients leave you.
  5. Saves a metrics table, the predicted probabilities (reused in later weeks),
     a reliability diagram, and a small settings file.

Predicted probabilities are clipped to [0.001, 0.999] before any metric is computed.
Random forests often output exactly 0 or 1, which would make log loss and the
calibration slope undefined.

Outputs (in results/):
  baseline_metrics.csv       one row per model
  baseline_predictions.csv   test patients: ID, y, probability from each model
  baseline_meta.json         settings, chosen hyper-parameters, counts
  figures/baseline_reliability.png

Usage (from the project folder, after python -m src.split):
  python -m src.baseline
  python -m src.baseline --seed 42 --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit, logit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.split import ID_COL, OUTCOME_COL, PROJECT_ROOT, TARGET_COL, load_dataset

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

CLIP = 1e-3
N_BINS = 10
MODEL_LABELS = {"logreg": "Logistic regression", "rf": "Random forest"}


def load_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (development pool, test set, feature column names), checked against split_meta.json."""
    pool_ids = pd.read_csv(SPLIT_DIR / "pool_ids.csv")[ID_COL]
    test_ids = pd.read_csv(SPLIT_DIR / "test_ids.csv")[ID_COL]
    meta = json.loads((SPLIT_DIR / "split_meta.json").read_text())

    assert not set(pool_ids) & set(test_ids), "Test and pool overlap"
    pool = df[df[ID_COL].isin(pool_ids)].sort_values(ID_COL).reset_index(drop=True)
    test = df[df[ID_COL].isin(test_ids)].sort_values(ID_COL).reset_index(drop=True)
    assert len(pool) == meta["n_pool"] and len(test) == meta["n_test"], "Split does not match split_meta.json"
    assert int(pool[OUTCOME_COL].sum()) == meta["n_events_pool"]
    assert int(test[OUTCOME_COL].sum()) == meta["n_events_test"]

    features = [c for c in df.columns if c not in (ID_COL, TARGET_COL, OUTCOME_COL)]
    assert len(features) == 30, f"Expected 30 features, found {len(features)}"
    return pool, test, features


def make_models(seed: int) -> dict[str, GridSearchCV]:
    """Both models, each wrapped so that its tuning setting is picked by cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    logreg = GridSearchCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),  # L2 penalty is the default
        param_grid={"logisticregression__C": np.logspace(-3, 2, 11)},
        scoring="neg_log_loss",
        cv=cv,
    )
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=500, random_state=seed, n_jobs=-1),
        param_grid={"min_samples_leaf": [1, 2, 5, 10]},
        scoring="neg_log_loss",
        cv=cv,
    )
    return {"logreg": logreg, "rf": rf}


def calibration_intercept(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration-in-the-large: the shift on the log-odds scale that makes the average
    predicted risk equal the observed rate. 0 is ideal; > 0 means risk is under-predicted."""
    lp = logit(p)
    return float(brentq(lambda a: expit(a + lp).mean() - y.mean(), -10, 10))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope from regressing the outcome on the predicted log-odds. 1 is ideal;
    < 1 means predictions are too extreme (over-confident)."""
    fit = LogisticRegression(C=1e6, max_iter=5000).fit(logit(p).reshape(-1, 1), y)
    return float(fit.coef_[0, 0])


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Group patients by predicted risk (equal-width bins); compare predicted with observed."""
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": p[m].mean(), "observed_rate": y[m].mean()})
    return pd.DataFrame(rows)


def expected_calibration_error(y: np.ndarray, p: np.ndarray) -> float:
    bins = reliability_bins(y, p)
    return float((bins["n"] * (bins["mean_predicted"] - bins["observed_rate"]).abs()).sum() / len(y))


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "auroc": roc_auc_score(y, p),
        "accuracy": accuracy_score(y, pred),
        "sensitivity": tp / int((y == 1).sum()),
        "specificity": tn / int((y == 0).sum()),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, p),
        "cal_intercept": calibration_intercept(y, p),
        "cal_slope": calibration_slope(y, p),
        "ece": expected_calibration_error(y, p),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n_boot: int, seed: int) -> dict[str, tuple[float, float]]:
    """95% percentile intervals from resampling the test patients with replacement."""
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {"auroc": [], "brier": [], "accuracy": []}
    n = len(y)
    while len(out["auroc"]) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():  # need both classes for AUROC
            continue
        out["auroc"].append(roc_auc_score(y[i], p[i]))
        out["brier"].append(brier_score_loss(y[i], p[i]))
        out["accuracy"].append(accuracy_score(y[i], (p[i] >= 0.5).astype(int)))
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def plot_reliability(y: np.ndarray, probs: dict[str, np.ndarray], path: Path) -> None:
    """Left: reliability diagram. Right: how many patients got each predicted risk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, axis = "#191919", "#666666", "#e7e7e7", "#c2c2c2"
    colours = {"logreg": "#257de0", "rf": "#f47230"}  # categorical slots 1 and 2

    fig, (ax, ax_h) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.patch.set_facecolor("white")
    for a in (ax, ax_h):
        a.set_facecolor("white")
        a.grid(color=grid, linewidth=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(axis)
        a.tick_params(colors=ink2, labelsize=9)

    ax.plot([0, 1], [0, 1], color=axis, linewidth=1.5, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.52, "perfect calibration", color=ink2, fontsize=8.5, rotation=36, ha="center")
    for key, p in probs.items():
        b = reliability_bins(y, p)
        ax.plot(b["mean_predicted"], b["observed_rate"], color=colours[key], linewidth=2, zorder=2)
        ax.scatter(b["mean_predicted"], b["observed_rate"], s=64, color=colours[key],
                   edgecolor="white", linewidth=1.5, zorder=3, label=MODEL_LABELS[key])
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Predicted probability of malignancy", color=ink2, fontsize=10)
    ax.set_ylabel("Observed share malignant", color=ink2, fontsize=10)
    ax.set_title("Reliability: do predicted risks match what happened?", loc="left", color=ink, fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5, labelcolor=ink)

    edges = np.linspace(0, 1, N_BINS + 1)
    width = 0.04
    for j, (key, p) in enumerate(probs.items()):
        counts, _ = np.histogram(p, bins=edges)
        centres = (edges[:-1] + edges[1:]) / 2 + (j - 0.5) * (width + 0.004)
        ax_h.bar(centres, counts, width=width, color=colours[key], label=MODEL_LABELS[key])
    ax_h.set_xlim(0, 1)
    ax_h.set_xlabel("Predicted probability of malignancy", color=ink2, fontsize=10)
    ax_h.set_ylabel("Number of test patients", color=ink2, fontsize=10)
    ax_h.set_title(f"Where the {len(y)} test predictions sit", loc="left", color=ink, fontsize=11, fontweight="bold")
    ax_h.legend(frameon=False, loc="upper center", fontsize=9.5, labelcolor=ink)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42)")
    parser.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples (default 1000)")
    args = parser.parse_args()

    df = load_dataset()
    pool, test, features = load_split(df)
    X_pool, y_pool = pool[features], pool[OUTCOME_COL].to_numpy()
    X_test, y_test = test[features], test[OUTCOME_COL].to_numpy()
    print(f"Development pool: {len(pool)} patients ({int(y_pool.sum())} malignant)")
    print(f"Test set:         {len(test)} patients ({int(y_test.sum())} malignant)\n")

    probs, rows, chosen = {}, [], {}
    for key, search in make_models(args.seed).items():
        search.fit(X_pool, y_pool)  # tuning happens inside the pool only
        p = np.clip(search.predict_proba(X_test)[:, 1], CLIP, 1 - CLIP)
        probs[key] = p
        chosen[key] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in search.best_params_.items()}
        res = evaluate(y_test, p)
        ci = bootstrap_ci(y_test, p, args.n_boot, args.seed)
        row = {"model": MODEL_LABELS[key], **res}
        for metric, (lo, hi) in ci.items():
            row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = lo, hi
        row["best_params"] = json.dumps(chosen[key])
        rows.append(row)

    metrics = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)
    pd.DataFrame({ID_COL: test[ID_COL], "y": y_test,
                  "p_logreg": probs["logreg"], "p_rf": probs["rf"]}).to_csv(
        RESULTS_DIR / "baseline_predictions.csv", index=False)
    plot_reliability(y_test, probs, FIG_DIR / "baseline_reliability.png")
    (RESULTS_DIR / "baseline_meta.json").write_text(json.dumps({
        "seed": args.seed, "n_boot": args.n_boot, "probability_clip": CLIP, "ece_bins": N_BINS,
        "n_pool": int(len(pool)), "n_events_pool": int(y_pool.sum()),
        "n_test": int(len(test)), "n_events_test": int(y_test.sum()),
        "chosen_hyperparameters": chosen,
    }, indent=2))

    show = ["auroc", "accuracy", "sensitivity", "specificity", "brier", "cal_intercept", "cal_slope", "ece"]
    print(metrics.set_index("model")[show].round(3).T.to_string())
    print()
    for r in rows:
        print(f"{r['model']}: AUROC {r['auroc']:.3f} (95% CI {r['auroc_ci_low']:.3f} to {r['auroc_ci_high']:.3f}), "
              f"Brier {r['brier']:.3f} ({r['brier_ci_low']:.3f} to {r['brier_ci_high']:.3f}), "
              f"chosen {r['best_params']}")
    print(f"\nWrote results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
