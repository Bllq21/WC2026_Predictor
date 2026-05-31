"""
Elo rating standalone para fútbol internacional.

Metodología basada en World Football Elo (eloratings.net), con dos
ajustes propios:
  1. K-factor variable por tipo de torneo (ya definido en TOURNAMENT_WEIGHTS)
  2. Margin of victory multiplier: ganar 3-0 sube más que ganar 1-0,
     pero con rendimientos decrecientes para evitar que goleadas históricas
     distorsionen el rating.

El resultado es un dict {equipo: elo_actual} que puede usarse como:
  - Feature standalone en predict.py (--elo flag)
  - Input de inicialización para otros modelos
  - Análisis exploratorio (quiénes son realmente los mejores ahora)

NO modifica Dixon-Coles, XGBoost, ni el bayesiano.
"""
import numpy as np
import pandas as pd

# K-factors por tipo de torneo.
# Valores calibrados empíricamente por World Football Elo y adoptados aquí.
ELO_K_FACTORS = {
    "FIFA World Cup": 60,
    "UEFA Euro": 50,
    "Copa América": 50,
    "African Cup of Nations": 50,
    "AFC Asian Cup": 50,
    "Gold Cup": 50,
    "CONCACAF Championship": 50,
    "Confederations Cup": 50,
    "Olympic Games": 50,
    "UEFA Nations League": 45,
    "CONCACAF Nations League": 45,
}
ELO_K_QUALIFIER = 40   # eliminatorias de cualquier torneo mayor
ELO_K_REGIONAL = 30    # torneos regionales sin clasificar como mayor
ELO_K_FRIENDLY = 20    # amistosos e invitacionales
ELO_DEFAULT_RATING = 1500  # rating inicial para todos los equipos


def get_k_factor(tournament):
    if tournament in ELO_K_FACTORS:
        return ELO_K_FACTORS[tournament]
    if "qualif" in tournament.lower():
        return ELO_K_QUALIFIER
    if tournament == "Friendly":
        return ELO_K_FRIENDLY
    return ELO_K_REGIONAL


def expected_score(rating_a, rating_b, home_advantage=100):
    """
    Probabilidad esperada de victoria para el equipo A.
    home_advantage=100 puntos Elo ~ +0.17 de probabilidad de victoria,
    valor estándar de World Football Elo para partidos en casa.
    Para cancha neutral, pasar home_advantage=0.
    """
    return 1 / (1 + 10 ** ((rating_b - (rating_a + home_advantage)) / 400))


def mov_multiplier(goal_diff, elo_diff):
    """
    Margin of Victory multiplier (estilo FiveThirtyEight/World Football Elo).
    Ajusta el K-factor por el margen de goles, con rendimientos decrecientes
    y corrección por la fuerza relativa de los equipos (para que ganar 3-0
    contra un equipo débil no sume tanto como ganar 3-0 contra uno fuerte).

    goal_diff: diferencia de goles del ganador (siempre positivo)
    elo_diff:  Elo del ganador - Elo del perdedor (puede ser negativo = upset)
    """
    return np.log(abs(goal_diff) + 1) * (2.2 / (elo_diff * 0.001 + 2.2))


def compute_elo(df, home_advantage=100, verbose=False):
    """
    Calcula los ratings Elo para todos los equipos procesando el dataset
    cronológicamente.

    Args:
        df: DataFrame de load_results() con columnas date_ordinal, home_team,
            away_team, home_score, away_score, tournament, neutral
        home_advantage: puntos Elo de ventaja para el equipo local.
                        Se ignora cuando neutral == True o 'TRUE'.
        verbose: si True, imprime los top 20 ratings al terminar.

    Returns:
        ratings: dict {equipo: elo_actual}
        history: DataFrame con la evolución de ratings partido a partido
                 (útil para graficar la trayectoria de un equipo)
    """
    df = df.sort_values("date_ordinal").reset_index(drop=True)
    ratings = {}
    history_rows = []

    for _, row in df.iterrows():
        home = row.home_team
        away = row.away_team
        hg = int(row.home_score)
        ag = int(row.away_score)
        is_neutral = str(row.get("neutral", "FALSE")).upper() == "TRUE"

        r_home = ratings.get(home, ELO_DEFAULT_RATING)
        r_away = ratings.get(away, ELO_DEFAULT_RATING)

        ha = 0 if is_neutral else home_advantage
        e_home = expected_score(r_home, r_away, ha)
        e_away = 1 - e_home

        k = get_k_factor(row.tournament)

        if hg > ag:
            s_home, s_away = 1, 0
            mov = mov_multiplier(hg - ag, (r_home + ha) - r_away)
        elif hg < ag:
            s_home, s_away = 0, 1
            mov = mov_multiplier(ag - hg, r_away - (r_home + ha))
        else:
            s_home, s_away = 0.5, 0.5
            mov = 1.0  # empate: sin multiplicador de margen

        ratings[home] = r_home + k * mov * (s_home - e_home)
        ratings[away] = r_away + k * mov * (s_away - e_away)

        history_rows.append({
            "date_str": row.date_str,
            "date_ordinal": row.date_ordinal,
            "home_team": home,
            "away_team": away,
            "home_elo_after": ratings[home],
            "away_elo_after": ratings[away],
            "tournament": row.tournament,
        })

    history = pd.DataFrame(history_rows)

    if verbose:
        top = sorted(ratings.items(), key=lambda x: -x[1])[:20]
        print("\nTop 20 ratings Elo actuales:")
        for i, (team, elo) in enumerate(top, 1):
            print(f"  {i:2d}. {team:<30} {elo:.0f}")

    return ratings, history


def predict_elo(ratings, home_team, away_team, neutral=True):
    """
    Predicción simple de resultado basada solo en Elo.
    Devuelve (P_home_win, P_draw, P_away_win).

    El Elo puro da P(home wins) como probabilidad continua, no distribuye
    empate/victoria explícitamente. Usamos el método de Dixon & Pope (2004):
    asumir que P(draw) depende de la cercanía de los ratings.
    """
    r_home = ratings.get(home_team, ELO_DEFAULT_RATING)
    r_away = ratings.get(away_team, ELO_DEFAULT_RATING)
    ha = 0 if neutral else 100

    p_home_or_draw = expected_score(r_home, r_away, ha)

    # Estimación de P(draw): mayor cuando los ratings son similares,
    # menor cuando hay gran diferencia. Calibrado empíricamente en ~0.22
    # promedio global (tasa real de empates en fútbol internacional).
    elo_diff = abs((r_home + ha) - r_away)
    p_draw = max(0.05, 0.22 - 0.001 * elo_diff)

    p_home = max(0, p_home_or_draw - p_draw / 2)
    p_away = max(0, 1 - p_home - p_draw)

    # Renormalizar para que sumen exactamente 1
    total = p_home + p_draw + p_away
    return p_home / total, p_draw / total, p_away / total


def get_team_elo(ratings, team):
    return ratings.get(team, ELO_DEFAULT_RATING)


if __name__ == "__main__":
    import sys
    from data_prep import load_results

    df = load_results()
    ratings, history = compute_elo(df, verbose=True)

    home = sys.argv[1] if len(sys.argv) > 1 else "Argentina"
    away = sys.argv[2] if len(sys.argv) > 2 else "France"

    print(f"\n{home} Elo: {ratings.get(home, ELO_DEFAULT_RATING):.0f}")
    print(f"{away} Elo: {ratings.get(away, ELO_DEFAULT_RATING):.0f}")
    print(f"Diferencia: {ratings.get(home, ELO_DEFAULT_RATING) - ratings.get(away, ELO_DEFAULT_RATING):+.0f}")

    p_h, p_d, p_a = predict_elo(ratings, home, away, neutral=True)
    print(f"\nPrediccion Elo (cancha neutral):")
    print(f"  {home} gana: {p_h:.1%}")
    print(f"  Empate:      {p_d:.1%}")
    print(f"  {away} gana: {p_a:.1%}")
