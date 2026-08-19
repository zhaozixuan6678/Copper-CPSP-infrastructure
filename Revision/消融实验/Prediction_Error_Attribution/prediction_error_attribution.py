#!/usr/bin/env python3
"""Four-stage prediction-error reduction attribution for HV, EC and Q3.

The three feature stages may live in different Excel files, sheets, column
layouts and feature sets.  All stages are aligned by normalized document ID
before one shared 80/20 split is generated for each property.

Stage S0: composition only
Stage S1: physicochemical feature engineering
Stage S2: CTD-derived CPSP features, same fixed model as S0/S1
Stage S3: same S2 data/features, model hyperparameters optimized by CV using
          training data only

Primary error: test NRMSE = RMSE(test) / population SD(y_test).
Attribution: dPhysics=E0-E1, dCTD=E1-E2, dModel=E2-E3, Residual=E3.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import sys
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

try:
    from catboost import CatBoostRegressor
except Exception:  # optional
    CatBoostRegressor = None


ALL_FEATURES = "__ALL_EXCEPT_ID_AND_TARGET__"


@dataclass
class StageData:
    stage: str
    label: str
    property_name: str
    target_column: str
    frame: pd.DataFrame
    features: list[str]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=here / "attribution_config.json")
    p.add_argument("--quick", action="store_true", help="Use 3 search draws for a fast pipeline check.")
    p.add_argument("--audit-only", action="store_true", help="Validate inputs/splits only; do not train models.")
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    required = {"random_seed", "test_size", "id_column", "stages", "targets"}
    missing = required.difference(cfg)
    if missing:
        raise ValueError(f"Config missing keys: {sorted(missing)}")
    return cfg


def normalize_id(value: Any) -> str | None:
    """Normalize Excel IDs such as 1173, 1173.0 and ' 1173 ' to one key."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        s = s.split(".", 1)[0]
    return s


def _target_aliases(cfg: dict[str, Any], prop: str) -> list[str]:
    defaults = {
        "HV": ["Hardness/HV", "HV"],
        "EC": ["EC/%IACS", "EC"],
        "Q3": ["Q3-Euclidean", "Q3-Eu"],
    }
    aliases = cfg.get("target_aliases", {}).get(prop, defaults[prop])
    return list(dict.fromkeys(aliases))


def auto_resolve_stage_inputs(cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Resolve S0/S1/S2 files from candidates using schema and target consistency.

    Resolution is deliberately conservative. A candidate must contain all three
    property sheets, a recognized target column, every explicitly configured
    feature, and any stage marker features. Candidate permutations are then
    validated through the same ID/target/cumulative-feature audit used for runs.
    Exactly one valid mapping is required; ambiguity is reported instead of guessed.
    """
    ac = cfg.get("auto_input_audit", {})
    candidates = [str(Path(p)) for p in ac.get("candidate_files", [])]
    if not candidates:
        raise ValueError("auto_input_audit.enabled=True but candidate_files is empty")
    candidates = list(dict.fromkeys(candidates))
    missing_paths = [p for p in candidates if not Path(p).exists()]
    if missing_paths:
        raise FileNotFoundError(f"Automatic audit candidate files do not exist: {missing_paths}")

    markers = ac.get("stage_required_markers", {})
    header_cache: dict[tuple[str, str], list[str]] = {}
    report_rows: list[dict[str, Any]] = []

    def columns_for(path: str, sheet: str) -> list[str]:
        key = (path, sheet)
        if key not in header_cache:
            try:
                header_cache[key] = list(pd.read_excel(path, sheet_name=sheet, nrows=0).columns)
            except Exception:
                header_cache[key] = []
        return header_cache[key]

    compatible: dict[tuple[str, str], dict[str, str]] = {}
    for stage in ("S0", "S1", "S2"):
        sc = cfg["stages"][stage]
        for path in candidates:
            target_map: dict[str, str] = {}
            reasons: list[str] = []
            for prop in cfg["targets"]:
                sheet = sc["sheets"][prop]
                cols = columns_for(path, sheet)
                if not cols:
                    reasons.append(f"{prop}: missing/unreadable sheet {sheet!r}")
                    continue
                found_targets = [name for name in _target_aliases(cfg, prop) if name in cols]
                if not found_targets:
                    reasons.append(f"{prop}: no target alias in columns")
                    continue
                target_map[prop] = found_targets[0]
                selected = sc["features"][prop]
                required = [] if selected == ALL_FEATURES else list(selected)
                required += list(markers.get(stage, {}).get(prop, []))
                missing = [name for name in dict.fromkeys(required) if name not in cols]
                if missing:
                    reasons.append(f"{prop}: missing features {missing}")
            passed = not reasons and len(target_map) == len(cfg["targets"])
            if passed:
                compatible[(stage, path)] = target_map
            report_rows.append({
                "Stage": stage,
                "Candidate_File": path,
                "Schema_Compatible": passed,
                "Detected_Targets": json.dumps(target_map, ensure_ascii=False, sort_keys=True),
                "Reason": "PASS" if passed else " | ".join(reasons),
            })

    valid_trials: list[dict[str, Any]] = []
    for assignment in permutations(candidates, 3):
        file_map = dict(zip(("S0", "S1", "S2"), assignment))
        if not all((stage, path) in compatible for stage, path in file_map.items()):
            continue
        trial = copy.deepcopy(cfg)
        trial["auto_input_audit"]["enabled"] = False
        for stage, path in file_map.items():
            trial["stages"][stage]["file"] = path
            trial["stages"][stage]["target_columns"] = compatible[(stage, path)]
        try:
            for prop in trial["targets"]:
                align_stages(trial, prop)
        except Exception as exc:
            report_rows.append({
                "Stage": "COMBINATION",
                "Candidate_File": json.dumps(file_map, ensure_ascii=False, sort_keys=True),
                "Schema_Compatible": False,
                "Detected_Targets": "",
                "Reason": f"cross-stage audit failed: {type(exc).__name__}: {exc}",
            })
            continue
        valid_trials.append(trial)

    report = pd.DataFrame(report_rows)
    if not valid_trials:
        concise = report[report["Schema_Compatible"]].to_dict("records")
        raise ValueError(
            "Automatic input audit found no valid S0/S1/S2 mapping. "
            f"Individually compatible candidates: {concise}"
        )
    if len(valid_trials) > 1:
        mappings = [
            {s: t["stages"][s]["file"] for s in ("S0", "S1", "S2")}
            for t in valid_trials
        ]
        raise ValueError(
            "Automatic input audit found multiple valid mappings; scientific role is ambiguous. "
            f"Restrict candidate_files or stage markers. Valid mappings: {mappings}"
        )
    return valid_trials[0], report


def read_stage(cfg: dict[str, Any], stage: str, prop: str) -> StageData:
    sc = cfg["stages"][stage]
    tc = sc.get("target_columns", {}).get(prop, cfg["targets"][prop]["column"])
    path = Path(sc["file"])
    sheet = sc["sheets"][prop]
    if not path.exists():
        raise FileNotFoundError(f"{stage}/{prop}: missing file: {path}")
    df = pd.read_excel(path, sheet_name=sheet, dtype=object)
    id_col = cfg["id_column"]
    for required in (id_col, tc):
        if required not in df.columns:
            raise KeyError(f"{stage}/{prop}: required column {required!r} not found in {path.name}/{sheet}")

    df = df.copy()
    df["__ID__"] = df[id_col].map(normalize_id)
    df["__TARGET__"] = pd.to_numeric(df[tc], errors="coerce")
    df = df[df["__ID__"].notna() & df["__TARGET__"].notna()].copy()
    if df["__ID__"].duplicated().any():
        dup = df.loc[df["__ID__"].duplicated(False), "__ID__"].unique()[:10]
        raise ValueError(f"{stage}/{prop}: duplicate valid IDs after normalization: {dup.tolist()}")

    selected = sc["features"][prop]
    if selected == ALL_FEATURES:
        excluded = {id_col, tc, "__ID__", "__TARGET__"}
        features = [str(c) for c in df.columns if c not in excluded]
    else:
        features = list(selected)
    missing_features = [c for c in features if c not in df.columns]
    if missing_features:
        raise KeyError(f"{stage}/{prop}: configured features absent from table: {missing_features}")

    # Numeric conversion is explicit. A fully nonnumeric selected column is an error.
    for col in features:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    all_nan = [c for c in features if df[c].notna().sum() == 0]
    if all_nan:
        raise ValueError(f"{stage}/{prop}: selected features contain no numeric data: {all_nan}")

    return StageData(stage, sc["label"], prop, tc, df, features)


def align_stages(cfg: dict[str, Any], prop: str) -> tuple[dict[str, StageData], pd.DataFrame]:
    data = {s: read_stage(cfg, s, prop) for s in ("S0", "S1", "S2")}
    common_ids = set.intersection(*(set(d.frame["__ID__"]) for d in data.values()))
    if not common_ids:
        raise ValueError(f"{prop}: no common normalized IDs across S0/S1/S2")

    ordered_ids = sorted(common_ids, key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
    target_matrix = pd.DataFrame(index=ordered_ids)
    for stage, d in data.items():
        indexed = d.frame.set_index("__ID__").loc[ordered_ids]
        d.frame = indexed.reset_index()
        target_matrix[stage] = indexed["__TARGET__"].astype(float)

    reference = target_matrix["S0"].to_numpy()
    target_diffs = {
        s: float(np.nanmax(np.abs(target_matrix[s].to_numpy() - reference)))
        for s in ("S1", "S2")
    }
    if cfg.get("strict_target_match", True) and any(v > 1e-10 for v in target_diffs.values()):
        raise ValueError(f"{prop}: target values differ across stages: {target_diffs}")
    target_matrix.insert(0, "ID", ordered_ids)

    # For a one-factor-at-a-time ablation, later stages must retain earlier
    # features.  The source tables may each contain only their own contribution,
    # so construct cumulative design matrices after ID alignment.  Duplicate
    # feature names are accepted only when their overlapping numeric values agree.
    if cfg.get("cumulative_features", True):
        cumulative = pd.DataFrame({"__ID__": ordered_ids, "__TARGET__": reference})
        cumulative_features: list[str] = []
        for stage in ("S0", "S1", "S2"):
            d = data[stage]
            current = d.frame.set_index("__ID__")
            cumulative = cumulative.set_index("__ID__")
            for feature in d.features:
                incoming = current[feature].astype(float).reindex(ordered_ids)
                if feature in cumulative.columns:
                    existing = cumulative[feature].astype(float)
                    mask = existing.notna() & incoming.notna()
                    if mask.any():
                        max_diff = float((existing[mask] - incoming[mask]).abs().max())
                        if max_diff > 1e-8:
                            raise ValueError(
                                f"{prop}/{stage}: duplicated feature {feature!r} differs "
                                f"across stage tables (max abs diff={max_diff:g})"
                            )
                    cumulative[feature] = existing.combine_first(incoming)
                else:
                    cumulative[feature] = incoming
                    cumulative_features.append(feature)
            cumulative = cumulative.reset_index()
            d.frame = cumulative.copy()
            d.features = cumulative_features.copy()
    return data, target_matrix


def split_ids(ids: list[str], test_size: float, seed: int) -> tuple[set[str], set[str]]:
    train_ids, test_ids = train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)
    return set(train_ids), set(test_ids)


def make_estimator(model_name: str, seed: int, params: dict[str, Any] | None = None):
    params = dict(params or {})
    name = model_name.upper()
    if name == "LGBM":
        base = dict(n_estimators=500, learning_rate=0.04, num_leaves=31,
                    subsample=0.9, colsample_bytree=0.9, random_state=seed,
                    n_jobs=-1, verbosity=-1)
        base.update(params)
        model = LGBMRegressor(**base)
    elif name == "XGB":
        base = dict(n_estimators=500, learning_rate=0.04, max_depth=5,
                    subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                    random_state=seed, n_jobs=-1, objective="reg:squarederror")
        base.update(params)
        model = XGBRegressor(**base)
    elif name == "CAT":
        if CatBoostRegressor is None:
            raise ImportError("catboost is required for model CAT")
        base = dict(iterations=500, learning_rate=0.04, depth=6,
                    random_seed=seed, verbose=False, allow_writing_files=False)
        base.update(params)
        model = CatBoostRegressor(**base)
    elif name == "ET":
        base = dict(n_estimators=500, min_samples_leaf=1, random_state=seed, n_jobs=-1)
        base.update(params)
        model = ExtraTreesRegressor(**base)
    elif name == "SVR":
        base = dict(C=10.0, epsilon=0.1, gamma="scale", kernel="rbf")
        base.update(params)
        model = SVR(**base)
    else:
        raise ValueError(f"Unsupported model {model_name!r}; use LGBM, XGB, CAT, ET or SVR")

    # Fit imputation using training data only. Scaling is harmless for trees and
    # necessary for SVR; keeping the same pipeline prevents preprocessing drift.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def search_space(model_name: str) -> dict[str, list[Any]]:
    name = model_name.upper()
    spaces = {
        "LGBM": {
            "model__n_estimators": [250, 400, 600, 800, 1000],
            "model__learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08],
            "model__num_leaves": [15, 23, 31, 47, 63],
            "model__max_depth": [-1, 4, 6, 8, 10],
            "model__min_child_samples": [5, 10, 20, 30, 50],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_lambda": [0.0, 0.1, 1.0, 5.0, 10.0],
        },
        "XGB": {
            "model__n_estimators": [250, 400, 600, 800, 1000],
            "model__learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08],
            "model__max_depth": [2, 3, 4, 5, 6, 8],
            "model__min_child_weight": [1, 2, 4, 6, 10],
            "model__subsample": [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
            "model__reg_alpha": [0.0, 0.01, 0.1, 0.5, 1.0],
            "model__reg_lambda": [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        "CAT": {
            "model__iterations": [250, 400, 600, 800, 1000],
            "model__learning_rate": [0.015, 0.025, 0.04, 0.06, 0.08],
            "model__depth": [4, 5, 6, 7, 8, 10],
            "model__l2_leaf_reg": [1, 3, 5, 8, 12],
        },
        "ET": {
            "model__n_estimators": [300, 500, 800, 1000],
            "model__max_depth": [None, 6, 10, 15, 25],
            "model__min_samples_split": [2, 4, 8, 12],
            "model__min_samples_leaf": [1, 2, 3, 5],
            "model__max_features": [0.5, 0.7, 0.9, 1.0],
        },
        "SVR": {
            "model__C": [0.5, 1, 3, 10, 30, 100],
            "model__epsilon": [0.01, 0.03, 0.05, 0.1, 0.2],
            "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1],
        },
    }
    return spaces[name]


def optuna_model_params(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    """Continuous/discrete search space used only inside the training set."""
    name = model_name.upper()
    if name != "XGB":
        raise ValueError("The current Optuna module is validated for XGB only")
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1800, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.1, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.60, 1.00),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.00),
        # Keep gamma at zero because HV, EC and Q3 have very different target
        # scales; sharing one absolute gamma range can suppress all Q3 splits.
        "gamma": 0.0,
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
        "max_bin": trial.suggest_int("max_bin", 128, 512, step=64),
        "tree_method": "hist",
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    sd = float(np.std(y_true, ddof=0))
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Target_SD_ddof0": sd,
        "NRMSE": rmse / sd if sd > 0 else np.nan,
    }


def fit_one(stage_data: StageData, model_name: str, train_ids: set[str], test_ids: set[str],
            seed: int, optimize: bool, n_iter: int, cv_folds: int,
            optimizer: str = "randomized", target_cv_r2: float | None = None):
    df = stage_data.frame.set_index("__ID__")
    train_order = [i for i in df.index if i in train_ids]
    test_order = [i for i in df.index if i in test_ids]
    Xtr = df.loc[train_order, stage_data.features]
    Xte = df.loc[test_order, stage_data.features]
    ytr = df.loc[train_order, "__TARGET__"].astype(float).to_numpy()
    yte = df.loc[test_order, "__TARGET__"].astype(float).to_numpy()

    estimator = make_estimator(model_name, seed)
    best_params: dict[str, Any] = {}
    cv_score = np.nan
    cv_r2 = np.nan
    if optimize:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        if optimizer.lower() == "optuna":
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
            study = optuna.create_study(direction="minimize", sampler=sampler)

            def objective(trial: optuna.Trial) -> float:
                params = optuna_model_params(trial, model_name)
                candidate = make_estimator(model_name, seed, params)
                scores = cross_validate(
                    candidate, Xtr, ytr, cv=cv,
                    scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                    n_jobs=1, return_train_score=False,
                )
                mean_rmse = float(-np.mean(scores["test_rmse"]))
                mean_r2 = float(np.mean(scores["test_r2"]))
                trial.set_user_attr("mean_cv_r2", mean_r2)
                trial.set_user_attr("std_cv_r2", float(np.std(scores["test_r2"], ddof=0)))
                trial.set_user_attr("mean_cv_rmse", mean_rmse)
                return mean_rmse

            def stop_at_training_target(study_: optuna.Study, trial_: optuna.Trial) -> None:
                if target_cv_r2 is not None and trial_.user_attrs.get("mean_cv_r2", -np.inf) >= target_cv_r2:
                    study_.stop()

            study.optimize(objective, n_trials=n_iter, callbacks=[stop_at_training_target], show_progress_bar=False)
            best_params = dict(study.best_trial.params)
            best_params["tree_method"] = "hist"
            cv_score = float(study.best_value)
            cv_r2 = float(study.best_trial.user_attrs["mean_cv_r2"])
            estimator = make_estimator(model_name, seed, best_params)
            estimator.fit(Xtr, ytr)
        elif optimizer.lower() == "randomized":
            search = RandomizedSearchCV(
                estimator, search_space(model_name), n_iter=n_iter,
                scoring="neg_root_mean_squared_error", cv=cv,
                random_state=seed, n_jobs=-1, refit=True, verbose=0,
            )
            search.fit(Xtr, ytr)
            estimator = search.best_estimator_
            best_params = search.best_params_
            cv_score = float(-search.best_score_)
        else:
            raise ValueError(f"Unknown Stage 3 optimizer: {optimizer!r}")
    else:
        estimator.fit(Xtr, ytr)

    pred_tr = estimator.predict(Xtr)
    pred_te = estimator.predict(Xte)
    mtr, mte = metrics(ytr, pred_tr), metrics(yte, pred_te)
    pred = pd.concat([
        pd.DataFrame({"ID": train_order, "Split": "Train", "y_true": ytr, "y_pred": pred_tr}),
        pd.DataFrame({"ID": test_order, "Split": "Test", "y_true": yte, "y_pred": pred_te}),
    ], ignore_index=True)
    return mtr, mte, pred, best_params, cv_score, cv_r2


def plot_attribution(attribution: pd.DataFrame, output: Path) -> None:
    components = ["Physics contribution", "CTD contribution", "Model contribution", "Residual error"]
    colors = ["#9AB6D3", "#D8876C", "#91B58A", "#6C7480"]
    fig, ax = plt.subplots(figsize=(7.3, 5.2), dpi=180)
    bottom = np.zeros(len(attribution))
    x = np.arange(len(attribution))
    for comp, color in zip(components, colors):
        vals = attribution[comp].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, width=0.62, label=comp, color=color,
               edgecolor="white", linewidth=0.8)
        bottom += vals
    ax.set_xticks(x, attribution["Property"])
    ax.set_ylabel("Test NRMSE (RMSE / SD of test target)")
    ax.set_title("Prediction Error Reduction Attribution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    auto_report: pd.DataFrame | None = None
    if cfg.get("auto_input_audit", {}).get("enabled", False):
        cfg, auto_report = auto_resolve_stage_inputs(cfg)
        resolved_path = args.config.with_name("auto_resolved_config.json")
        with resolved_path.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        print("Automatic input audit resolved a unique mapping:")
        for stage in ("S0", "S1", "S2"):
            print(
                f"  {stage}: {cfg['stages'][stage]['file']} | "
                f"targets={cfg['stages'][stage]['target_columns']}"
            )
        print(f"Resolved configuration: {resolved_path}")
    output = Path(cfg["output_directory"])
    if args.quick:
        output = output.with_name(output.name + "_quick")
    output.mkdir(parents=True, exist_ok=True)
    if auto_report is not None:
        auto_report.to_csv(output / "auto_input_audit_report.csv", index=False, encoding="utf-8-sig")
    seed = int(cfg["random_seed"])

    all_audit, all_splits = [], []
    prepared: dict[str, tuple[dict[str, StageData], set[str], set[str]]] = {}
    for prop in cfg["targets"]:
        stage_data, targets = align_stages(cfg, prop)
        ids = targets["ID"].tolist()
        train_ids, test_ids = split_ids(ids, float(cfg["test_size"]), seed)
        prepared[prop] = (stage_data, train_ids, test_ids)
        for stage, d in stage_data.items():
            all_audit.append({
                "Property": prop, "Stage": stage, "Stage_Label": d.label,
                "Source_File": cfg["stages"][stage]["file"],
                "Source_Sheet": cfg["stages"][stage]["sheets"][prop],
                "Source_Target_Column": d.target_column,
                "Common_Valid_IDs": len(ids), "Feature_Count": len(d.features),
                "Features": " | ".join(d.features),
            })
        all_splits.extend({"Property": prop, "ID": i, "Split": "Train" if i in train_ids else "Test", "Seed": seed} for i in ids)

    audit_df = pd.DataFrame(all_audit)
    split_df = pd.DataFrame(all_splits)
    audit_df.to_csv(output / "input_audit.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(output / "split_map.csv", index=False, encoding="utf-8-sig")
    print("Input audit passed. Common IDs and feature counts:")
    print(audit_df[["Property", "Stage", "Common_Valid_IDs", "Feature_Count"]].to_string(index=False))
    if args.audit_only:
        return 0

    if args.quick:
        n_iter = 3
    elif str(cfg.get("stage3_optimizer", "randomized")).lower() == "optuna":
        n_iter = int(cfg.get("optuna_trials", 150))
    else:
        n_iter = int(cfg.get("stage3_search_iterations", 30))
    summary_rows, prediction_frames, best_param_rows = [], [], []
    for prop, tc in cfg["targets"].items():
        model_name = tc["model"]
        stage_data, train_ids, test_ids = prepared[prop]
        for stage in ("S0", "S1", "S2", "S3"):
            source_stage = "S2" if stage == "S3" else stage
            optimize = stage == "S3"
            d = stage_data[source_stage]
            print(f"Running {prop} {stage} with {model_name}; optimize={optimize}")
            mtr, mte, pred, params, cv_rmse, cv_r2 = fit_one(
                d, model_name, train_ids, test_ids, seed, optimize, n_iter,
                int(cfg.get("cv_folds", 5)),
                optimizer=str(cfg.get("stage3_optimizer", "randomized")),
                target_cv_r2=cfg.get("optuna_target_cv_r2"),
            )
            run_id = f"{prop}_{stage}_{model_name}_seed{seed}"
            summary_rows.append({
                "Run_ID": run_id, "Property": prop, "Stage": stage,
                "Feature_Source_Stage": source_stage, "Model": model_name,
                "Optimized": optimize, "Seed": seed,
                "n_train": len(train_ids), "n_test": len(test_ids),
                "Feature_Count": len(d.features), "Features": " | ".join(d.features),
                "Train_R2": mtr["R2"], "Test_R2": mte["R2"],
                "Train_RMSE": mtr["RMSE"], "Test_RMSE": mte["RMSE"],
                "Test_MAE": mte["MAE"], "Test_Target_SD_ddof0": mte["Target_SD_ddof0"],
                "Test_NRMSE": mte["NRMSE"], "CV_RMSE_training_only": cv_rmse,
                "CV_R2_training_only": cv_r2,
            })
            pred.insert(0, "Run_ID", run_id)
            pred.insert(1, "Property", prop)
            pred.insert(2, "Stage", stage)
            pred.insert(3, "Model", model_name)
            prediction_frames.append(pred)
            best_param_rows.append({"Run_ID": run_id, "Best_Params_JSON": json.dumps(params, ensure_ascii=False, sort_keys=True)})

    summary = pd.DataFrame(summary_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    best_params = pd.DataFrame(best_param_rows)

    attribution_rows = []
    for prop in cfg["targets"]:
        sub = summary[summary["Property"] == prop].set_index("Stage")
        e0, e1, e2, e3 = (float(sub.loc[s, "Test_NRMSE"]) for s in ("S0", "S1", "S2", "S3"))
        vals = [e0 - e1, e1 - e2, e2 - e3, e3]
        attribution_rows.append({
            "Property": prop, "E0": e0, "E1": e1, "E2": e2, "E3": e3,
            "Physics contribution": vals[0], "CTD contribution": vals[1],
            "Model contribution": vals[2], "Residual error": vals[3],
            "Physics_%_of_E0": vals[0] / e0 * 100,
            "CTD_%_of_E0": vals[1] / e0 * 100,
            "Model_%_of_E0": vals[2] / e0 * 100,
            "Residual_%_of_E0": vals[3] / e0 * 100,
            "Telescoping_Check": sum(vals) - e0,
            "Negative_Component_Flag": any(v < 0 for v in vals[:3]),
        })
    attribution = pd.DataFrame(attribution_rows)

    summary.to_csv(output / "run_summary.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(output / "predictions_long.csv", index=False, encoding="utf-8-sig")
    best_params.to_csv(output / "best_parameters.csv", index=False, encoding="utf-8-sig")
    attribution.to_csv(output / "attribution_summary.csv", index=False, encoding="utf-8-sig")
    plot_attribution(attribution, output / "prediction_error_attribution.png")

    print("\nPrediction error attribution (test NRMSE):")
    print(attribution.to_string(index=False))
    print(f"\nSaved outputs to: {output}")
    if attribution["Negative_Component_Flag"].any():
        print("WARNING: at least one sequential stage increased error. Keep the signed value; do not force it to zero.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
