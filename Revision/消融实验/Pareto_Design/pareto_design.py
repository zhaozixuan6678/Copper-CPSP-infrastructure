"""Constrained Pareto discovery for Cu-Ni-Si-Cr alloys.

The script retrains the locked XGBoost models, samples a large composition space,
recalculates every engineered feature, applies an applicability-domain filter,
builds a robust HV-EC Pareto front and exports three representative alloys.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ELEMENTS = ["Cu", "Al", "Cr", "Mg", "Ni", "Si", "Zr"]
ATOMIC_MASS = {"Cu": 63.546, "Al": 26.982, "Cr": 51.996, "Mg": 24.305,
               "Ni": 58.693, "Si": 28.085, "Zr": 91.224}
ATOMIC_RADIUS = {"Cu": 1.28, "Al": 1.43, "Cr": 1.28, "Mg": 1.60,
                 "Ni": 1.24, "Si": 1.11, "Zr": 1.60}
ELECTRONEGATIVITY = {"Cu": 1.90, "Al": 1.61, "Cr": 1.66, "Mg": 1.31,
                     "Ni": 1.91, "Si": 1.90, "Zr": 1.33}
VEC = {"Cu": 11, "Al": 3, "Cr": 6, "Mg": 2, "Ni": 10, "Si": 4, "Zr": 4}
MELTING_POINT = {"Cu": 1084.62, "Al": 660.32, "Cr": 1907.0, "Mg": 650.0,
                 "Ni": 1455.0, "Si": 1414.0, "Zr": 1855.0}

SHEET_MAP = {"HV": "HV", "EC": "EC", "Q3": "Q3-Eu"}
TARGET_MAP = {"HV": "HV", "EC": "EC", "Q3": "Q3-Eu"}


DEFAULT_CONFIG = {
    "feature_file": "/Users/zixuanzhao/Desktop/Corpus–Topic–Document/FE/Feature/Feature3.xlsx",
    "raw_file": "/Users/zixuanzhao/Desktop/Corpus–Topic–Document/FE/Feature1.xlsx",
    "model_file": "/Users/zixuanzhao/Desktop/Corpus–Topic–Document/ML2/XGB.xlsx",
    "output_dir": "/Users/zixuanzhao/Desktop/Corpus–Topic–Document/ML2/消融实验/Pareto_Design/results",
    "random_seed": 42,
    "test_size": 0.20,
    "n_candidates": 131072,
    "bootstrap_models": 8,
    "confidence_z": 1.0,
    "ad_quantile": 0.95,
    "ad_neighbors": 5,
    "fit_full_for_design": True,
    "composition_ranges_wt_pct": {
        "Al": [0.30, 0.50], "Cr": [0.10, 0.30], "Mg": [0.01, 0.16],
        "Ni": [4.00, 6.00], "Si": [1.00, 1.40], "Zr": [0.00, 0.00]
    },
    "composition_rounding_wt_pct": {
        "Al": 0.01, "Cr": 0.01, "Mg": 0.01, "Ni": 0.02, "Si": 0.01, "Zr": 0.01
    },
    "constraints": {"cu_min_wt_pct": 92.0, "ni_si_ratio": [3.5, 6.0]},
    "fixed_processing": {
        "Processing_Route": 2, "Solution_Temp": 980.0, "Solution_Time": 4.0,
        "CR_Reduction": 50.0, "Aging_Temp": 450.0, "Aging_Time": 4.0
    },
    "experimental_alloy": None
}


def load_config(path: str | None) -> dict:
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            user = json.load(fh)
        for key, value in user.items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    return cfg


def parse_model_book(path: str) -> dict:
    df = pd.read_excel(path, sheet_name="BestParams")
    out = {}
    for key, sheet in SHEET_MAP.items():
        row = df.loc[df["Sheet"].astype(str).eq(sheet)].iloc[0]
        features = row["selected_features"]
        if isinstance(features, str):
            features = ast.literal_eval(features) if features.strip().startswith("[") else [x.strip() for x in features.split("+")]
        params = ast.literal_eval(str(row["final_params"]))
        params["random_state"] = 42
        params["n_jobs"] = -1
        out[key] = {"features": list(features), "params": params}
    return out


def _descriptor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for el in ELEMENTS:
        if el not in out:
            out[el] = 0.0
    non_cu = [e for e in ELEMENTS if e != "Cu"]
    out["Cu"] = 100.0 - out[non_cu].sum(axis=1)
    mole = np.column_stack([out[e].to_numpy(float) / ATOMIC_MASS[e] for e in ELEMENTS])
    xi = mole / mole.sum(axis=1, keepdims=True)
    arr_r = np.array([ATOMIC_RADIUS[e] for e in ELEMENTS])
    arr_chi = np.array([ELECTRONEGATIVITY[e] for e in ELEMENTS])
    arr_vec = np.array([VEC[e] for e in ELEMENTS])
    arr_tm = np.array([MELTING_POINT[e] for e in ELEMENTS])
    r_avg = xi @ arr_r
    chi_avg = xi @ arr_chi
    vec_avg = xi @ arr_vec
    out["Tmavg"] = xi @ arr_tm
    out["VECavg"] = vec_avg
    out["VECvar"] = xi @ (arr_vec ** 2) - vec_avg ** 2
    out["δ"] = np.sqrt(np.sum(xi * (1 - arr_r / r_avg[:, None]) ** 2, axis=1))
    out["χvar"] = np.sum(xi * (arr_chi - chi_avg[:, None]) ** 2, axis=1)
    out["Smix"] = -8.314 * np.sum(np.where(xi > 0, xi * np.log(np.clip(xi, 1e-15, None)), 0), axis=1)
    out["HHI"] = np.sum(xi ** 2, axis=1)
    out["N_eff"] = 1.0 / out["HHI"]
    # Preserve the definition used in the stored feature table (mole fraction,
    # not percentage points): e.g. 0.101759 rather than 10.1759.
    out["Alloying"] = 1.0 - xi[:, 0]
    out["Ni/wt.%"] = out["Ni"]
    out["Si/wt.%"] = out["Si"]
    out["Ni/Si"] = out["Ni"] / out["Si"].replace(0, np.nan)
    out["Ni*Si"] = out["Ni"] * out["Si"]
    out["Mg/(Ni+Si)"] = out["Mg"] / (out["Ni"] + out["Si"]).replace(0, np.nan)
    out["Family"] = np.where((out["Cr"] > 0) & (out["Ni"] > 0) & (out["Si"] > 0), 2,
                             np.where(out["Cr"] > 0, 1, 0))
    out["One-Hot-Processing"] = pd.to_numeric(out["Processing_Route"], errors="coerce").fillna(0).astype(int)
    out["CR_x_ATem"] = out["CR_Reduction"] / 100.0 * out["Aging_Temp"]
    out["ATem_x_Atime"] = out["Aging_Temp"] * out["Aging_Time"]
    out["STem-ATem"] = out["Solution_Temp"] - out["Aging_Temp"]
    return out


def generate_candidates(cfg: dict) -> pd.DataFrame:
    ranges = cfg["composition_ranges_wt_pct"]
    variable = [k for k, (lo, hi) in ranges.items() if hi > lo]
    fixed = {k: lo for k, (lo, hi) in ranges.items() if hi <= lo}
    n = int(cfg["n_candidates"])
    sampler = qmc.Sobol(d=len(variable), scramble=True, seed=int(cfg["random_seed"]))
    m = int(math.ceil(math.log2(max(2, n))))
    unit = sampler.random_base2(m=m)[:n]
    lo = np.array([ranges[k][0] for k in variable])
    hi = np.array([ranges[k][1] for k in variable])
    values = qmc.scale(unit, lo, hi)
    df = pd.DataFrame(values, columns=variable)
    for key, val in fixed.items():
        df[key] = val
    for key, step in cfg["composition_rounding_wt_pct"].items():
        if step and key in df:
            df[key] = np.round(df[key] / step) * step
    df = df.drop_duplicates().reset_index(drop=True)
    for key, value in cfg["fixed_processing"].items():
        df[key] = value
    df["Cu"] = 100.0 - df[[e for e in ELEMENTS if e != "Cu"]].sum(axis=1)
    ratio = df["Ni"] / df["Si"]
    c = cfg["constraints"]
    keep = (df["Cu"] >= c["cu_min_wt_pct"]) & ratio.between(*c["ni_si_ratio"])
    return _descriptor_matrix(df.loc[keep].reset_index(drop=True))


def load_training(cfg: dict, model_cfg: dict) -> tuple[dict, pd.DataFrame]:
    data = {}
    for key in ["HV", "EC", "Q3"]:
        df = pd.read_excel(cfg["feature_file"], sheet_name=SHEET_MAP[key])
        cols = ["ID"] + model_cfg[key]["features"] + [TARGET_MAP[key]]
        # The locked modeling workflow median-imputes missing predictors.  Only
        # rows without an ID or target are invalid for supervised fitting.
        data[key] = df[cols].dropna(subset=["ID", TARGET_MAP[key]]).copy()
    raw = pd.read_excel(cfg["raw_file"], sheet_name="HV")
    return data, raw


def audit_feature_formulas(candidates_like_raw: pd.DataFrame, feature_data: dict, out_dir: Path) -> pd.DataFrame:
    # Rebuild features for real rows and compare with stored values.
    raw = candidates_like_raw.rename(columns={
        "Cu/wt.%": "Cu", "Al/wt.%": "Al", "Cr/wt.%": "Cr", "Mg/wt.%": "Mg",
        "Ni/wt.%": "Ni", "Si/wt.%": "Si", "Zr/wt.%": "Zr",
        "One-Hot-Processing": "Processing_Route", "Solution_Temperature/℃": "Solution_Temp",
        "Solution_Time/h": "Solution_Time", "CR_Reduction/%": "CR_Reduction",
        "Aging_Temperature/℃": "Aging_Temp", "Aging_Time/h": "Aging_Time"
    }).copy()
    rebuilt = _descriptor_matrix(raw)
    rows = []
    for key, stored in feature_data.items():
        merged = stored.merge(rebuilt, on="ID", suffixes=("_stored", "_calc"))
        for feat in [f for f in stored.columns if f not in {"ID", TARGET_MAP[key]}]:
            a = pd.to_numeric(merged[f"{feat}_stored"], errors="coerce")
            b = pd.to_numeric(merged[f"{feat}_calc"], errors="coerce")
            d = (a - b).abs()
            rows.append({"Property": key, "Feature": feat, "N": int(d.notna().sum()),
                         "MAE_formula": float(d.mean()), "Max_abs_error": float(d.max())})
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "feature_formula_audit.csv", index=False)
    return result


def make_pipeline(params: dict, seed: int) -> Pipeline:
    p = dict(params)
    p["random_state"] = seed
    p["n_jobs"] = -1
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("model", XGBRegressor(**p))])


def train_and_predict(cfg: dict, model_cfg: dict, data: dict, candidates: pd.DataFrame, out_dir: Path):
    rng = np.random.default_rng(int(cfg["random_seed"]))
    predictions = {}
    audits = []
    for key in ["HV", "EC", "Q3"]:
        df = data[key]
        feats, target = model_cfg[key]["features"], TARGET_MAP[key]
        ids = df["ID"].to_numpy()
        tr_ids, te_ids = train_test_split(ids, test_size=cfg["test_size"], random_state=cfg["random_seed"])
        train = df[df["ID"].isin(tr_ids)]
        test = df[df["ID"].isin(te_ids)]
        audit_model = make_pipeline(model_cfg[key]["params"], int(cfg["random_seed"]))
        audit_model.fit(train[feats], train[target])
        for split, part in [("Train", train), ("Test", test)]:
            pred = audit_model.predict(part[feats])
            audits.append({"Property": key, "Split": split, "N": len(part),
                           "R2": r2_score(part[target], pred),
                           "RMSE": mean_squared_error(part[target], pred) ** 0.5})
        fit_df = df if cfg["fit_full_for_design"] else train
        boot_preds = []
        for b in range(int(cfg["bootstrap_models"])):
            if b == 0:
                boot = fit_df
            else:
                boot = fit_df.iloc[rng.integers(0, len(fit_df), len(fit_df))]
            model = make_pipeline(model_cfg[key]["params"], int(cfg["random_seed"]) + b)
            model.fit(boot[feats], boot[target])
            boot_preds.append(model.predict(candidates[feats]))
        arr = np.vstack(boot_preds)
        predictions[key] = {"mean": arr.mean(axis=0), "std": arr.std(axis=0, ddof=1) if len(arr) > 1 else np.zeros(arr.shape[1])}
    audit_df = pd.DataFrame(audits)
    audit_df.to_csv(out_dir / "model_reproduction_audit.csv", index=False)
    return predictions, audit_df


def applicability_domain(cfg: dict, raw: pd.DataFrame, candidates: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    cols_train = ["Al/wt.%", "Cr/wt.%", "Mg/wt.%", "Ni/wt.%", "Si/wt.%", "Zr/wt.%",
                  "Solution_Temperature/℃", "CR_Reduction/%", "Aging_Temperature/℃", "Aging_Time/h"]
    cols_cand = ["Al", "Cr", "Mg", "Ni", "Si", "Zr", "Solution_Temp", "CR_Reduction", "Aging_Temp", "Aging_Time"]
    train = raw[(raw["Cr/wt.%"] > 0) & (raw["Ni/wt.%"] > 0) & (raw["Si/wt.%"] > 0)][cols_train].dropna()
    scaler = StandardScaler().fit(train.to_numpy(float))
    z_train = scaler.transform(train.to_numpy(float))
    z_cand = scaler.transform(candidates[cols_cand].to_numpy(float))
    k = int(cfg["ad_neighbors"])
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z_train)
    train_dist = nn.kneighbors(z_train, return_distance=True)[0][:, -1]
    threshold = float(np.quantile(train_dist, cfg["ad_quantile"]))
    cand_dist = NearestNeighbors(n_neighbors=k).fit(z_train).kneighbors(z_cand, return_distance=True)[0][:, -1]
    return cand_dist <= threshold, cand_dist, threshold


def pareto_mask(hv: np.ndarray, ec: np.ndarray) -> np.ndarray:
    order = np.lexsort((-ec, -hv))
    keep = np.zeros(len(hv), dtype=bool)
    best_ec = -np.inf
    for idx in order:
        if ec[idx] > best_ec:
            keep[idx] = True
            best_ec = ec[idx]
    return keep


def select_three(front: pd.DataFrame) -> pd.DataFrame:
    strength_i = front["HV_LCB"].idxmax()
    conductivity_i = front["EC_LCB"].idxmax()
    h = front["HV_LCB"]
    e = front["EC_LCB"]
    hn = (h - h.min()) / max(h.max() - h.min(), 1e-12)
    en = (e - e.min()) / max(e.max() - e.min(), 1e-12)
    distance = np.sqrt((1 - hn) ** 2 + (1 - en) ** 2)
    knee_i = distance.idxmin()
    chosen = front.loc[[strength_i, knee_i, conductivity_i]].copy()
    chosen.insert(0, "Pareto_Point", ["P1_Strength_anchor", "P2_Balanced_knee", "P3_Conductivity_anchor"])
    chosen["Utopia_distance"] = distance.loc[[strength_i, knee_i, conductivity_i]].to_numpy()
    chosen["Recommended_experimental_role"] = ["Boundary/reference", "Primary experimental alloy", "Boundary/reference"]
    return chosen.drop_duplicates(subset=["Pareto_Point"])


def plot_results(all_df: pd.DataFrame, front: pd.DataFrame, points: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=180)
    valid = all_df[all_df["AD_valid"]]
    sample = valid.sample(min(30000, len(valid)), random_state=42) if len(valid) else valid
    ax.scatter(
        sample["EC_mean"], sample["HV_mean"],
        s=15, c="#9fadb9", alpha=.55,
        edgecolors="white", linewidths=.22,
        label="In-domain candidates", zorder=1
    )
    f = front.sort_values("EC_LCB")
    ax.plot(f["EC_mean"], f["HV_mean"], color="#c96532", lw=2.2, label="Robust Pareto front")
    colors = ["#9a3d2f", "#3f8f68", "#3f6f9f"]
    for (_, row), color in zip(points.iterrows(), colors):
        ax.scatter(row["EC_mean"], row["HV_mean"], s=75, c=color, edgecolor="white", linewidth=1.2, zorder=5)
        ax.annotate(row["Pareto_Point"].split("_", 1)[0], (row["EC_mean"], row["HV_mean"]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Predicted electrical conductivity (%IACS)")
    ax.set_ylabel("Predicted hardness (HV)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_front.png", transparent=True, bbox_inches="tight")
    fig.savefig(out_dir / "pareto_front.pdf", transparent=True, bbox_inches="tight")
    plt.close(fig)


def main(config_path: str | None = None):
    cfg = load_config(config_path)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    model_cfg = parse_model_book(cfg["model_file"])
    data, raw = load_training(cfg, model_cfg)
    formula_audit = audit_feature_formulas(raw, data, out_dir)
    candidates = generate_candidates(cfg)
    ad_valid, ad_distance, ad_threshold = applicability_domain(cfg, raw, candidates)
    pred, model_audit = train_and_predict(cfg, model_cfg, data, candidates, out_dir)
    result = candidates[["Cu", "Al", "Cr", "Mg", "Ni", "Si", "Zr", "Processing_Route",
                         "Solution_Temp", "Solution_Time", "CR_Reduction", "Aging_Temp", "Aging_Time"]].copy()
    result["AD_valid"] = ad_valid
    result["AD_distance"] = ad_distance
    z = float(cfg["confidence_z"])
    for key in ["HV", "EC", "Q3"]:
        result[f"{key}_mean"] = pred[key]["mean"]
        result[f"{key}_std"] = pred[key]["std"]
        result[f"{key}_LCB"] = pred[key]["mean"] - z * pred[key]["std"]
    valid = result[result["AD_valid"]].copy()
    mask = pareto_mask(valid["HV_LCB"].to_numpy(), valid["EC_LCB"].to_numpy())
    front = valid.loc[mask].copy().sort_values("EC_LCB").reset_index(drop=True)
    points = select_three(front)
    result.to_csv(out_dir / "all_candidates.csv.gz", index=False, compression="gzip")
    front.to_csv(out_dir / "pareto_front.csv", index=False)
    points.to_csv(out_dir / "pareto_three_points.csv", index=False)
    search_rows = [{"Variable": k, "Minimum": v[0], "Maximum": v[1], "Unit": "wt.%"} for k, v in cfg["composition_ranges_wt_pct"].items()]
    pd.DataFrame(search_rows).to_csv(out_dir / "search_space.csv", index=False)
    plot_results(result, front, points, out_dir)
    summary = {
        "generated_candidates": int(len(result)), "in_domain_candidates": int(result["AD_valid"].sum()),
        "pareto_points": int(len(front)), "ad_threshold": ad_threshold,
        "formula_audit_median_mae": float(formula_audit["MAE_formula"].median()),
        "output_dir": str(out_dir)
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nThree representative Pareto points:\n", points.to_string(index=False))
    print("\nModel reproduction audit:\n", model_audit.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    main(args.config)
