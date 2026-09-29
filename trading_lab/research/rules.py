"""Baselines intraday déterministes, causales et fixées avant le test final."""
import numpy as np
import pandas as pd

RULE_IDS = ["RULE_MOMENTUM", "RULE_VWAP_RECLAIM", "RULE_ORB20"]


def _binary(mask, index, name):
    # 1.0 = signal ; 0.0 = absence de signal. Compatible avec le seuil V9 figé.
    return pd.Series(np.where(mask, 1.0, 0.0), index=index, name=name, dtype=float)


def rule_predictions(bars, features, index):
    b = bars.loc[index]
    f = features.loc[index]
    progress = f["SessionProgress"]
    liquid = f["RelativeVolumeSlot"].fillna(0) >= 0.8
    tradable_time = progress.between(0.08, 0.82)

    momentum = (f["Return3"] > 0) & (f["Return10"] > 0) & (f["DistVWAP"] > 0) & liquid & tradable_time

    prior_dist = features["DistVWAP"].shift(1).reindex(index)
    prior_session = bars["Session"].shift(1).reindex(index)
    same_session = prior_session.eq(b["Session"])
    vwap_reclaim = same_session & (prior_dist <= 0) & (f["DistVWAP"] > 0) & (f["SessionReturn"] < 0.006) & liquid & tradable_time

    opening_high = bars.groupby("Session")["High"].transform(lambda s: s.iloc[:4].max())
    after_or = progress >= (20 / 390)
    # Le niveau des 20 premières minutes n'est utilisé qu'une fois ces 4 bougies closes.
    orb = after_or & (b["Close"] > opening_high.reindex(index)) & (f["DistVWAP"] > 0) & liquid & (progress <= 0.75)

    return pd.concat([
        _binary(momentum, index, "RULE_MOMENTUM"),
        _binary(vwap_reclaim, index, "RULE_VWAP_RECLAIM"),
        _binary(orb, index, "RULE_ORB20"),
    ], axis=1)
