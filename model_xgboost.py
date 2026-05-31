"""
Modelo XGBoost (Groll et al. 2019): dos regresores de goles esperados
(uno para goles del local, otro para el visitante) usando features de
fuerza ofensiva/defensiva derivadas del historial reciente de cada equipo,
luego se construye una matriz de Poisson con esos goles esperados.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import poisson

from data_prep import load_results, add_decay_weight


def build_team_features(df, half_life_days=730):
    """
    Para cada partido, calcula la fuerza ofensiva/defensiva de cada equipo
    usando SOLO partidos anteriores a esa fecha (evita data leakage).
    Implementación vectorizada: por equipo, un EWM (exponentially weighted
    mean) de goles marcados/recibidos, desplazado un partido (shift) para
    no incluir el resultado del partido actual.
    """
    df = df.sort_values("date_ordinal").reset_index(drop=True)
    df["match_id"] = df.index

    # Vista "larga": una fila por equipo por partido (home y away apiladas)
    home_view = df[["match_id", "date_ordinal", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "goals_for", "away_score": "goals_against"})
    home_view["is_home"] = 1
    away_view = df[["match_id", "date_ordinal", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "goals_for", "home_score": "goals_against"})
    away_view["is_home"] = 0

    long_df = pd.concat([home_view, away_view], ignore_index=True)
    long_df = long_df.sort_values(["team", "date_ordinal", "match_id"]).reset_index(drop=True)

    # EWM con span equivalente al half_life, calculado POR EQUIPO,
    # y desplazado 1 para que el feature de un partido no use su propio resultado.
    alpha = 1 - 0.5 ** (1 / (half_life_days / 30))  # aproximación mensual razonable
    grp = long_df.groupby("team")
    long_df["attack_strength"] = grp["goals_for"].transform(
        lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())
    long_df["defense_strength"] = grp["goals_against"].transform(
        lambda s: s.shift(1).ewm(alpha=alpha, adjust=False).mean())

    # Default para el primer partido de un equipo (sin historial previo)
    long_df["attack_strength"] = long_df["attack_strength"].fillna(1.3)
    long_df["defense_strength"] = long_df["defense_strength"].fillna(1.3)

    home_feats = long_df[long_df.is_home == 1][["match_id", "attack_strength", "defense_strength"]]
    home_feats.columns = ["match_id", "home_attack", "home_defense"]
    away_feats = long_df[long_df.is_home == 0][["match_id", "attack_strength", "defense_strength"]]
    away_feats.columns = ["match_id", "away_attack", "away_defense"]

    out = df.merge(home_feats, on="match_id").merge(away_feats, on="match_id")
    out["neutral_flag"] = (out["neutral"] == True) | (out["neutral"] == "TRUE")
    out["neutral"] = out["neutral_flag"].astype(int)
    return out[["home_team", "away_team", "date_ordinal", "home_attack", "home_defense",
                "away_attack", "away_defense", "home_score", "away_score", "neutral"]]


class XGBoostGoalModel:
    FEATURES = ["home_attack", "home_defense", "away_attack", "away_defense", "neutral"]

    def __init__(self):
        self.model_home = None
        self.model_away = None
        self.team_strength = {}  # cache de fuerza actual por equipo

    def fit(self, feat_df):
        train = feat_df.dropna(subset=["home_score", "away_score"])
        X = train[self.FEATURES]
        y_home = train["home_score"]
        y_away = train["away_score"]

        params = dict(
            objective="count:poisson",  # clave: regresión de Poisson, no MSE
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        self.model_home = xgb.XGBRegressor(**params).fit(X, y_home)
        self.model_away = xgb.XGBRegressor(**params).fit(X, y_away)
        return self

    def current_team_strength(self, feat_df, team):
        """Última fuerza conocida de un equipo (attack/defense), del feature set."""
        rows = feat_df[(feat_df.home_team == team) | (feat_df.away_team == team)]
        if rows.empty:
            return 1.3, 1.3
        last = rows.sort_values("date_ordinal").iloc[-1]
        if last.home_team == team:
            return last.home_attack, last.home_defense
        return last.away_attack, last.away_defense

    def predict_lambdas(self, feat_df, home_team, away_team, neutral=0):
        h_atk, h_def = self.current_team_strength(feat_df, home_team)
        a_atk, a_def = self.current_team_strength(feat_df, away_team)
        X = pd.DataFrame([{
            "home_attack": h_atk, "home_defense": h_def,
            "away_attack": a_atk, "away_defense": a_def, "neutral": neutral,
        }])
        lambda_home = float(self.model_home.predict(X)[0])
        lambda_away = float(self.model_away.predict(X)[0])
        return max(lambda_home, 0.05), max(lambda_away, 0.05)

    def predict_scoreline_matrix(self, feat_df, home_team, away_team, max_goals=10, neutral=0):
        lh, la = self.predict_lambdas(feat_df, home_team, away_team, neutral)
        h_range = np.arange(max_goals + 1)
        matrix = np.outer(poisson.pmf(h_range, lh), poisson.pmf(h_range, la))
        matrix /= matrix.sum()
        return matrix

    def predict_outcome(self, feat_df, home_team, away_team, max_goals=10, neutral=0):
        m = self.predict_scoreline_matrix(feat_df, home_team, away_team, max_goals, neutral)
        p_home = np.tril(m, -1).sum()
        p_draw = np.trace(m)
        p_away = np.triu(m, 1).sum()
        return p_home, p_draw, p_away

    def top_scorelines(self, feat_df, home_team, away_team, n=10, max_goals=10, neutral=0):
        m = self.predict_scoreline_matrix(feat_df, home_team, away_team, max_goals, neutral)
        flat = [(h, a, m[h, a]) for h in range(max_goals + 1) for a in range(max_goals + 1)]
        flat.sort(key=lambda x: -x[2])
        return flat[:n]


if __name__ == "__main__":
    df = load_results()  # ya excluye fixtures futuros sin jugar (dropna interno)
    print("Construyendo features (esto recorre todo el histórico, puede tardar)...")
    feat_df = build_team_features(df)

    print("Entrenando XGBoost...")
    model = XGBoostGoalModel().fit(feat_df)

    p_home, p_draw, p_away = model.predict_outcome(feat_df, "Argentina", "Austria", neutral=1)
    print(f"\nArgentina vs Austria (cancha neutral):")
    print(f"  Argentina gana: {p_home:.1%}")
    print(f"  Empate:         {p_draw:.1%}")
    print(f"  Austria gana:   {p_away:.1%}")

    print("\nTop 5 marcadores:")
    for h, a, p in model.top_scorelines(feat_df, "Argentina", "Austria", n=5, neutral=1):
        print(f"  {h}-{a}: {p:.1%}")
