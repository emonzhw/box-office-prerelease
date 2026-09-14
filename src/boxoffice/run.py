import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from boxoffice.data import attach_gdp, clean_movies, load_gdp, load_tmdb
from boxoffice.evaluate import TEMPORAL_CUTOFF, cv_predictions, cv_r2, in_sample_adj_r2, models, temporal_r2
from boxoffice.features import build_matrix

FEATURE_SETS = ["replication", "full", "pre_release", "budget_only"]


def evaluate_all(movies):
    y = movies["log_revenue"].to_numpy()
    years = movies["release_year"].to_numpy()
    rows = []
    for fs in FEATURE_SETS:
        X = build_matrix(movies, fs)
        for name, model in models().items():
            if fs in ("replication", "budget_only") and name != "ols":
                continue
            mean, std = cv_r2(model, X.to_numpy(), y)
            rows.append(
                {
                    "feature_set": fs,
                    "model": name,
                    "n_features": X.shape[1],
                    "in_sample_adj_r2": in_sample_adj_r2(X, y) if name == "ols" else float("nan"),
                    "cv_r2_mean": mean,
                    "cv_r2_std": std,
                    f"temporal_r2_train_pre{TEMPORAL_CUTOFF}": temporal_r2(model, X.to_numpy(), y, years),
                }
            )
    return pd.DataFrame(rows)


def plot_r2(metrics, out):
    temporal = f"temporal_r2_train_pre{TEMPORAL_CUTOFF}"
    plot = metrics[metrics["model"] == "ols"].set_index("feature_set").loc[["replication", "full", "pre_release", "budget_only"]]
    labels = ["team model", "full", "pre-release", "budget only"]
    x = range(len(plot))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar([i - 0.2 for i in x], plot["cv_r2_mean"], width=0.4, yerr=plot["cv_r2_std"], capsize=3, color="#8a9ba8", label="5-fold CV")
    ax.bar([i + 0.2 for i in x], plot[temporal], width=0.4, color="#1f4d5c", label=f"train <{TEMPORAL_CUTOFF}, test ≥{TEMPORAL_CUTOFF}")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Out-of-sample R², log revenue (OLS)")
    ax.set_title("Most of the accuracy comes from information you don't have before release")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_pred_vs_actual(movies, out):
    X = build_matrix(movies, "pre_release").to_numpy()
    y = movies["log_revenue"].to_numpy()
    pred = cv_predictions(models()["ols"], X, y)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, pred, s=6, alpha=0.3, color="#1f4d5c")
    lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], color="#9c2f45", linewidth=1)
    ax.set_xlabel("Actual log revenue")
    ax.set_ylabel("Predicted (pre-release features, CV)")
    ax.set_title("Pre-release forecast (OLS, 5-fold CV)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare box-office models with and without post-release features.")
    p.add_argument("--movies", default="data/raw/tmdb_5000_movies.csv")
    p.add_argument("--gdp", default="data/raw/us-gdp.csv")
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    movies = attach_gdp(clean_movies(load_tmdb(args.movies)), load_gdp(args.gdp))
    metrics = evaluate_all(movies)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "metrics.md").write_text(f"n = {len(movies)} films\n\n" + metrics.round(3).to_markdown(index=False) + "\n")
    plot_r2(metrics, out / "r2_comparison.png")
    plot_pred_vs_actual(movies, out / "pre_release_pred_vs_actual.png")
    print(f"n = {len(movies)}")
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
