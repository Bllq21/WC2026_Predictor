"""
Evaluación del modelo bayesiano sobre un holdout MÁS CHICO que el de
evaluate.py, porque cada partido evaluado requiere su propio entrenamiento
MCMC acotado a los equipos relevantes. Es lento por diseño (MCMC > MLE),
así que se corre aparte y con un N bajo por default.
"""
import sys
import numpy as np
import pandas as pd

from data_prep import load_results, add_decay_weight
from model_bayesian import BayesianGoalModel
from evaluate import rps, actual_outcome_idx


def evaluate_bayesian(df, n_holdout=20, max_teams=40, draws=500, tune=500, chains=2):
    df = df.sort_values("date_ordinal").reset_index(drop=True)  # load_results() ya excluye nulos
    holdout_df = df.tail(n_holdout).copy()
    train_df_full = df.iloc[:-n_holdout].copy()

    accs, rpss = [], []
    for i, (_, row) in enumerate(holdout_df.iterrows()):
        train_df = add_decay_weight(train_df_full, half_life_days=730, reference_date=row.date_str)
        model = BayesianGoalModel(draws=draws, tune=tune, chains=chains)
        try:
            model.fit(train_df, relevant_teams=[row.home_team, row.away_team], max_teams=max_teams)
        except KeyError:
            continue  # equipo sin historial suficiente en el subset

        summary = model.predict_outcome(row.home_team, row.away_team)
        probs = [summary["home"]["mean"], summary["draw"]["mean"], summary["away"]["mean"]]
        pred_idx = int(np.argmax(probs))
        actual_idx = actual_outcome_idx(row.home_score, row.away_score)
        accs.append(pred_idx == actual_idx)
        rpss.append(rps(probs, actual_idx))
        print(f"  [{i+1}/{len(holdout_df)}] {row.home_team} vs {row.away_team}: "
              f"pred={['H','D','A'][pred_idx]} actual={['H','D','A'][actual_idx]}")

    return np.mean(accs), np.mean(rpss), len(accs)


if __name__ == "__main__":
    n_holdout = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    df = load_results()
    print(f"Evaluando modelo bayesiano sobre {n_holdout} partidos "
          f"(cada uno re-entrena MCMC, esto es LENTO a propósito)...")
    acc, rps_score, n = evaluate_bayesian(df, n_holdout=n_holdout)
    print(f"\nAccuracy: {acc:.1%}  RPS: {rps_score:.4f}  (evaluados: {n})")
