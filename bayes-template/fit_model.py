import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pymc as pm
    import arviz as az
except Exception as exc:
    print("ERROR: PyMC/ArviZ not installed. Please install requirements.txt first.")
    print(str(exc))
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bayesian longitudinal state-space model (log-normal observation)"
    )
    parser.add_argument("--input", required=True, help="CSV file or folder with CSV files")
    parser.add_argument("--output", default="posterior_draws.json", help="Output JSON path")
    parser.add_argument("--id", default="id", dest="id_col", help="ID column name")
    parser.add_argument("--group", default="group", dest="group_col", help="Group column name")
    parser.add_argument("--time", default="time", dest="time_col", help="Time column name")
    parser.add_argument("--y", default="y", dest="y_col", help="Measurement column name")
    parser.add_argument("--draws", type=int, default=1000, help="Posterior draws per chain")
    parser.add_argument("--tune", type=int, default=1000, help="Tuning steps per chain")
    parser.add_argument("--chains", type=int, default=4, help="Number of MCMC chains")
    parser.add_argument("--target-accept", type=float, default=0.9, help="NUTS target_accept")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--min-ess", type=int, default=200, help="Minimum ESS threshold")
    parser.add_argument("--max-draws", type=int, default=4000, help="Max total draws stored in output")
    return parser.parse_args()


def load_data(path_str):
    path = Path(path_str)
    if path.is_dir():
        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
        frames = [pd.read_csv(p) for p in csv_files]
        data = pd.concat(frames, ignore_index=True)
    else:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        data = pd.read_csv(path)
    return data


def validate_data(df, id_col, group_col, time_col, y_col):
    for col in [id_col, group_col, time_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if (df[y_col] <= 0).any():
        bad_rows = df[df[y_col] <= 0].head(5)
        raise ValueError(
            "y must be positive for LogNormal. Found y <= 0. "
            f"Example rows:\n{bad_rows}"
        )
    df = df.copy()
    df[time_col] = df[time_col].astype(int)
    return df


def build_indices(df, group_col, time_col):
    groups = sorted(df[group_col].unique().tolist())
    times = sorted(df[time_col].unique().tolist())
    group_to_idx = {g: i for i, g in enumerate(groups)}
    time_to_idx = {t: i for i, t in enumerate(times)}
    group_idx = df[group_col].map(group_to_idx).to_numpy()
    time_idx = df[time_col].map(time_to_idx).to_numpy()
    return groups, times, group_idx, time_idx


def main():
    args = parse_args()
    df = load_data(args.input)
    df = validate_data(df, args.id_col, args.group_col, args.time_col, args.y_col)

    groups, times, group_idx, time_idx = build_indices(df, args.group_col, args.time_col)
    n_groups = len(groups)
    n_times = len(times)

    y = df[args.y_col].to_numpy()
    logy = np.log(y)

    time_index = np.arange(n_times)
    time_index = time_index.reshape(-1, 1)

    y_mean = float(np.mean(y))
    if y_mean <= 0:
        y_mean = 1.0

    with pm.Model() as model:
        # Weakly informative priors
        mu_drift = pm.Normal("mu_drift", mu=0.0, sigma=0.5)
        sigma_drift = pm.HalfNormal("sigma_drift", sigma=0.5)
        drift = pm.Normal("drift", mu=mu_drift, sigma=sigma_drift, shape=n_groups)

        sigma_rw = pm.HalfNormal("sigma_rw", sigma=0.3)
        sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.5)

        logA0 = pm.Normal("logA0", mu=np.log(y_mean), sigma=1.0, shape=n_groups)

        rw = pm.Normal("rw", mu=0.0, sigma=sigma_rw, shape=(n_times - 1, n_groups))
        rw_cum = pm.math.cumsum(rw, axis=0)
        zeros = pm.math.zeros((1, n_groups))
        rw_full = pm.math.concatenate([zeros, rw_cum], axis=0)

        logA = pm.Deterministic(
            "logA",
            logA0 + drift * time_index + rw_full,
        )

        pm.Normal("logy", mu=logA[time_idx, group_idx], sigma=sigma_obs, observed=logy)

        trace = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            random_seed=args.seed,
            progressbar=True,
        )

    summary = az.summary(trace, var_names=["mu_drift", "sigma_drift", "sigma_rw", "sigma_obs", "drift"])
    bad_rhat = summary[summary["r_hat"] > 1.01]
    bad_ess = summary[summary["ess_bulk"] < args.min_ess]
    if not bad_rhat.empty or not bad_ess.empty:
        print("ERROR: Sampling diagnostics are poor. Consider increasing tune/draws or target_accept.")
        if not bad_rhat.empty:
            print("High R-hat:")
            print(bad_rhat[["r_hat"]])
        if not bad_ess.empty:
            print("Low ESS:")
            print(bad_ess[["ess_bulk"]])
        sys.exit(2)

    logA_draws = trace.posterior["logA"].values
    # shape: (chains, draws, time, group) -> (samples, time, group)
    logA_draws = logA_draws.reshape(-1, n_times, n_groups)
    total_draws = logA_draws.shape[0]

    if total_draws > args.max_draws:
        step = max(1, int(np.floor(total_draws / args.max_draws)))
        logA_draws = logA_draws[::step]

    A_draws = np.exp(logA_draws)

    payload = {
        "groups": groups,
        "times": times,
        "draws": A_draws.tolist(),
        "meta": {
            "n_groups": n_groups,
            "n_times": n_times,
            "n_draws": int(A_draws.shape[0]),
            "columns": {
                "id": args.id_col,
                "group": args.group_col,
                "time": args.time_col,
                "y": args.y_col,
            },
            "model": "log-normal observation + random walk with group drift",
        },
    }

    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    print(f"Saved posterior draws to: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
