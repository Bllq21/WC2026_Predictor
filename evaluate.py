"""
Evaluación de Dixon-Coles y XGBoost sobre un holdout temporal (los N partidos
más recientes, nunca usados para entrenar). Métricas: Accuracy 1X2 y RPS
(Rank Probability Score - mejor métrica que accuracy para outcomes
ordinales como 1X2, porque penaliza menos un error "cercano" como predecir
empate cuando ganó el local, que un error "lejano" como predecir derrota
cuando ganó el local).
"""
import numpy as np
import pandas as pd

from data_prep import load_results, add_decay_weight
from model_dixon_coles import DixonColes
from model_xgboost import build_team_features, XGBoostGoalModel


def rps(probs, actual_outcome_idx, n_outcomes=3):
    """
    Rank Probability Score para un solo partido.
    probs: [P(home), P(draw), P(away)]
    actual_outcome_idx: 0=home gana, 1=empate, 2=away gana
    Menor = mejor. 0 = predicción perfecta.
    """
    cum_probs = np.cumsum(probs)
    cum_actual = np.zeros(n_outcomes)
    cum_actual[actual_outcome_idx:] = 1
    return np.sum((cum_probs - cum_actual) ** 2) / (n_outcomes - 1)


def actual_outcome_idx(home_score, away_score):
    if home_score > away_score:
        return 0
    elif home_score == away_score:
        return 1
    return 2


def evaluate_dixon_coles(train_df, holdout_df):
    model = DixonColes().fit(train_df)
    accs, rpss = [], []
    skipped = 0
    for _, row in holdout_df.iterrows():
        if row.home_team not in model.team_idx or row.away_team not in model.team_idx:
            skipped += 1
            continue
        p_home, p_draw, p_away = model.predict_outcome(row.home_team, row.away_team)
        pred_idx = np.argmax([p_home, p_draw, p_away])
        actual_idx = actual_outcome_idx(row.home_score, row.away_score)
        accs.append(pred_idx == actual_idx)
        rpss.append(rps([p_home, p_draw, p_away], actual_idx))
    return np.mean(accs), np.mean(rpss), skipped, len(accs)


def evaluate_xgboost(train_df, holdout_df):
    feat_df = build_team_features(train_df)
    model = XGBoostGoalModel().fit(feat_df)

    # Para predecir en el holdout, necesitamos la fuerza de cada equipo
    # actualizada con TODO el historial de entrenamiento (no leakage:
    # el holdout es posterior en el tiempo, así que esto es válido).
    full_feat_df = build_team_features(pd.concat([train_df, holdout_df]).sort_values("date_ordinal"))
    known_teams = set(train_df.home_team) | set(train_df.away_team)

    accs, rpss = [], []
    skipped = 0
    for _, row in holdout_df.iterrows():
        if row.home_team not in known_teams or row.away_team not in known_teams:
            skipped += 1
            continue
        neutral = 1 if row.get("neutral") in (True, "TRUE") else 0
        p_home, p_draw, p_away = model.predict_outcome(
            full_feat_df[full_feat_df.date_ordinal < row.date_ordinal], row.home_team, row.away_team, neutral=neutral)
        pred_idx = np.argmax([p_home, p_draw, p_away])
        actual_idx = actual_outcome_idx(row.home_score, row.away_score)
        accs.append(pred_idx == actual_idx)
        rpss.append(rps([p_home, p_draw, p_away], actual_idx))
    return np.mean(accs), np.mean(rpss), skipped, len(accs)


if __name__ == "__main__":
    import sys
    N_HOLDOUT = int(sys.argv[1]) if len(sys.argv) > 1 else 200  # últimos N partidos

    df = load_results().sort_values("date_ordinal").reset_index(drop=True)  # ya sin nulos

    holdout_df = df.tail(N_HOLDOUT).copy()
    train_df = df.iloc[:-N_HOLDOUT].copy()
    train_df = add_decay_weight(train_df, half_life_days=730)

    print(f"Train: {len(train_df)} partidos | Holdout: {len(holdout_df)} partidos "
          f"({holdout_df.date_str.min()} a {holdout_df.date_str.max()})")

    print("\n--- Dixon-Coles ---")
    acc, rps_score, skipped, n = evaluate_dixon_coles(train_df, holdout_df)
    print(f"Accuracy: {acc:.1%}  RPS: {rps_score:.4f}  (evaluados: {n}, omitidos: {skipped})")

    print("\n--- XGBoost ---")
    acc, rps_score, skipped, n = evaluate_xgboost(train_df, holdout_df)
    print(f"Accuracy: {acc:.1%}  RPS: {rps_score:.4f}  (evaluados: {n}, omitidos: {skipped})")

    print("\n(Modelo bayesiano omitido del benchmark automático: requiere acotar")
    print(" el universo de equipos por partido, ver model_bayesian.py. Se evalúa")
    print(" aparte, sobre un subconjunto, en evaluate_bayesian.py)")
