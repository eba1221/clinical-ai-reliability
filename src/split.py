"""Create the data splits for the sample-size experiment.

Dataset: UCI Breast Cancer Wisconsin (Diagnostic), id=17 (569 patients, 30 features).

The script produces:
  1. One fixed, stratified held-out test set. Every model, at every training size,
     is evaluated on these same patients, so differences in discrimination and
     calibration come from training size, not from a changing test set.
  2. A development pool (everything not in the test set) from which training sets
     of increasing size are drawn, repeated many times with different seeds.

Training subsamples are stratified (outcome prevalence matches the pool) and nested
within each repeat: for repeat r, the size-50 sample contains the size-25 sample,
the size-100 sample contains the size-50 sample, and so on. Nesting means that
within a repeat, the only thing changing between sizes is the added patients,
which reduces noise when comparing sizes.

Outputs (in --out-dir, default data/splits/):
  test_ids.csv       patient IDs in the held-out test set
  pool_ids.csv       patient IDs in the development pool
  train_subsets.csv  long format: repeat, train_size, id
  split_meta.json    settings and class counts, for the methods section

The raw dataset is cached to data/raw/wdbc.csv on first run, so later runs work
offline and always use the same copy of the data.

Usage:
  python -m src.split
  python -m src.split --test-size 0.3 --sizes 25 50 100 200 --repeats 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "wdbc.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "splits"

ID_COL = "ID"
TARGET_COL = "Diagnosis"  # 'M' (malignant) / 'B' (benign) in the source data
OUTCOME_COL = "y"  # 1 = malignant, 0 = benign

DEFAULT_SIZES = [25, 50, 75, 100, 150, 200, 300] # the nested sets


def load_dataset(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    """Return one row per patient: ID, 30 features, Diagnosis, and binary outcome y."""
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=17)
        df = pd.concat([ds.data.ids, ds.data.features, ds.data.targets], axis=1)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)

    if df[ID_COL].duplicated().any():
        raise ValueError("Duplicate patient IDs in the dataset")
    df[OUTCOME_COL] = (df[TARGET_COL] == "M").astype(int)
    return df


def nested_stratified_subsets(
    ids: np.ndarray, y: np.ndarray, sizes: list[int], rng: np.random.Generator
) -> dict[int, np.ndarray]:
    """Draw nested, stratified subsamples of `ids` for each size in `sizes`.

    Positives and negatives are shuffled separately once; each size takes the first
    k positives and first n-k negatives, so smaller samples are subsets of larger ones.
    """
    pos = rng.permutation(ids[y == 1])
    neg = rng.permutation(ids[y == 0])
    prevalence = len(pos) / len(ids)

    subsets = {}
    for n in sorted(sizes):
        n_pos = int(round(n * prevalence))
        n_pos = min(max(n_pos, 1), len(pos), n - 1)  # keep at least one of each class
        n_neg = n - n_pos
        if n_neg > len(neg):
            raise ValueError(f"Size {n} is larger than the pool allows")
        subsets[n] = np.concatenate([pos[:n_pos], neg[:n_neg]])
    return subsets


def make_splits(
    df: pd.DataFrame, test_size: float, sizes: list[int], repeats: int, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pool_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[OUTCOME_COL], random_state=seed
    )

    pool_n = len(pool_df)
    bad = [n for n in sizes if n > pool_n or n < 2]
    if bad:
        raise ValueError(f"Training sizes {bad} must be between 2 and the pool size ({pool_n})")

    pool_ids = pool_df[ID_COL].to_numpy()
    pool_y = pool_df[OUTCOME_COL].to_numpy()

    # One independent random stream per repeat, derived from the master seed, so a
    # repeat's subsets do not change if --repeats is increased later.
    children = np.random.SeedSequence(seed).spawn(repeats)
    rows = []
    for r, child in enumerate(children):
        rng = np.random.default_rng(child)
        for n, ids in nested_stratified_subsets(pool_ids, pool_y, sizes, rng).items():
            rows.append(pd.DataFrame({"repeat": r, "train_size": n, ID_COL: ids}))
    subsets = pd.concat(rows, ignore_index=True)

    test_out = test_df[[ID_COL, OUTCOME_COL]].sort_values(ID_COL).reset_index(drop=True)
    pool_out = pool_df[[ID_COL, OUTCOME_COL]].sort_values(ID_COL).reset_index(drop=True)
    return test_out, pool_out, subsets


def check_splits(test: pd.DataFrame, pool: pd.DataFrame, subsets: pd.DataFrame) -> None:# ensures that
    """Fail loudly if any split is leaky or malformed."""
    test_ids = set(test[ID_COL])
    pool_ids = set(pool[ID_COL])
    assert not test_ids & pool_ids, "Test and pool overlap"
    assert set(subsets[ID_COL]) <= pool_ids, "Training subset contains non-pool IDs"

    for (r, n), g in subsets.groupby(["repeat", "train_size"]):
        assert len(g) == n, f"Repeat {r}, size {n}: got {len(g)} rows"
        assert g[ID_COL].is_unique, f"Repeat {r}, size {n}: duplicate IDs"

    for r, g in subsets.groupby("repeat"):
        by_size = {n: set(x[ID_COL]) for n, x in g.groupby("train_size")}
        ordered = sorted(by_size)
        for small, big in zip(ordered, ordered[1:]):
            assert by_size[small] <= by_size[big], f"Repeat {r}: size {small} not nested in {big}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-size", type=float, default=0.3, help="Fraction held out for testing (default 0.3)")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES, help="Training sample sizes")
    parser.add_argument("--repeats", type=int, default=100, help="Subsamples per training size (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default 42)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    df = load_dataset()
    test, pool, subsets = make_splits(df, args.test_size, args.sizes, args.repeats, args.seed)
    check_splits(test, pool, subsets)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    test[[ID_COL]].to_csv(args.out_dir / "test_ids.csv", index=False)
    pool[[ID_COL]].to_csv(args.out_dir / "pool_ids.csv", index=False)
    subsets.to_csv(args.out_dir / "train_subsets.csv", index=False)

    y_lookup = df.set_index(ID_COL)[OUTCOME_COL]
    size_summary = (
        subsets.assign(y=subsets[ID_COL].map(y_lookup))
        .groupby("train_size")["y"]
        .agg(n_events_per_sample=lambda s: s.sum() / args.repeats)
        .reset_index()
    )
    meta = {
        "dataset": "UCI Breast Cancer Wisconsin (Diagnostic), id=17",
        "outcome": "y = 1 if Diagnosis == 'M' (malignant)",
        "seed": args.seed,
        "test_size": args.test_size,
        "repeats": args.repeats,
        "train_sizes": sorted(args.sizes),
        "n_total": int(len(df)),
        "n_events_total": int(df[OUTCOME_COL].sum()),
        "n_test": int(len(test)),
        "n_events_test": int(test[OUTCOME_COL].sum()),
        "n_pool": int(len(pool)),
        "n_events_pool": int(pool[OUTCOME_COL].sum()),
        "events_per_train_size": {
            int(r.train_size): float(r.n_events_per_sample) for r in size_summary.itertuples()
        },
    }
    (args.out_dir / "split_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Total {meta['n_total']} ({meta['n_events_total']} malignant)")
    print(f"Test  {meta['n_test']} ({meta['n_events_test']} malignant)")
    print(f"Pool  {meta['n_pool']} ({meta['n_events_pool']} malignant)")
    print(f"{args.repeats} repeats x sizes {sorted(args.sizes)}")
    for n, e in meta["events_per_train_size"].items():
        print(f"  n={n:>4}: {e:.0f} events")
    print(f"Wrote splits to {args.out_dir}")


if __name__ == "__main__":
    main()
