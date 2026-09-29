"""Modèles V9 CPU : RF/GB + régression logistique, sans split aléatoire.

Les sorties sont des scores de modèle. Elles ne sont jamais présentées comme des
probabilités validées ou calibrées.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from threadpoolctl import threadpool_limits
from ..features.engine import GROUPS

TREE_MODEL_IDS = [f"{family}_{group}" for family in ("RF", "GB") for group in GROUPS]
LINEAR_MODEL_IDS = ["LR_A", "LR_D"]
MODEL_IDS = TREE_MODEL_IDS + LINEAR_MODEL_IDS


def make_model(family, cfg):
    if family == "RF":
        return RandomForestClassifier(n_estimators=cfg.rf_trees, max_depth=5,
                                      min_samples_leaf=20, max_features="sqrt",
                                      n_jobs=cfg.jobs, random_state=cfg.seed,
                                      class_weight="balanced_subsample")
    if family == "GB":
        return HistGradientBoostingClassifier(max_iter=cfg.gb_iterations, max_leaf_nodes=15,
                                             min_samples_leaf=20, l2_regularization=1,
                                             learning_rate=0.05, early_stopping=False,
                                             random_state=cfg.seed)
    if family == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.3, class_weight="balanced", max_iter=1000,
                                                random_state=cfg.seed))
    raise ValueError(f"Famille inconnue : {family}")


def eligible_train(bars, features, labels, boundary, timeframe, cfg):
    first = bars.index.searchsorted(boundary, side="left")
    gap = cfg.horizon_minutes // timeframe + 1 + cfg.extra_embargo_bars
    ready = features[GROUPS["D"]].notna().all(axis=1)
    mask = (np.arange(len(bars)) < first - gap) & ready & labels.Target.notna() & (labels.LabelEnd < boundary)
    if int(mask.sum()) < cfg.min_train_rows or labels.loc[mask, "Target"].nunique() < 2:
        raise ValueError(f"Entraînement insuffisant : {int(mask.sum())} lignes étiquetées, deux classes requises.")
    last = int(np.flatnonzero(mask)[-1])
    audit = dict(train_rows=int(mask.sum()), train_last=bars.index[last].isoformat(),
                 train_label_end_max=labels.loc[mask, "LabelEnd"].max().isoformat(),
                 prediction_boundary=boundary.isoformat(), gap_bars=first - last - 1,
                 required_gap_bars=gap, leakage_check=True)
    assert labels.loc[mask, "LabelEnd"].max() < boundary
    assert first - last - 1 >= gap
    return mask, audit


def fit_predict(model_id, features, labels, mask, prediction_index, cfg):
    family, group = model_id.split("_", 1)
    columns = GROUPS[group]
    model = make_model(family, cfg)
    with threadpool_limits(limits=cfg.jobs):
        model.fit(features.loc[mask, columns], labels.loc[mask, "Target"].astype(int))
        score = model.predict_proba(features.loc[prediction_index, columns])[:, 1]
    return pd.Series(score, index=prediction_index, name=model_id)
