"""
Predicción de penales: capa adicional encima de los modelos de 90'/120'.
NO modifica Dixon-Coles, XGBoost, ni el bayesiano.

Metodología:
- Tasa histórica de victoria en penales por selección (de shootouts.csv)
- Corrección por ventaja del primer pateador (53.1% históricamente, no 60%
  como se cita popularmente - calculado sobre 256 partidos con dato)
- Suavizado bayesiano simple: equipos con pocos datos heredan parcialmente
  la tasa global promedio (~51%), en vez de reportar 0% o 100% con 3 partidos
- El output es la P(gana en penales) condicional a llegar empatados a penales

Uso: se llama DESPUÉS de los modelos de 90' para dar la predicción completa
de un partido de eliminación directa (16avos, cuartos, etc.).
"""
import pandas as pd
import numpy as np


GLOBAL_PENALTY_WIN_RATE = 0.511  # media global empírica sobre todos los equipos
FIRST_SHOOTER_ADVANTAGE = 0.531  # quién patea primero gana 53.1% (256 partidos)
SMOOTHING_PRIOR = 10             # peso del prior global (equivale a 10 "partidos virtuales")


def load_shootout_stats(path="shootouts.csv"):
    """
    Carga el historial de penales y calcula la tasa de victoria por equipo
    con suavizado bayesiano (para equipos con pocos datos).

    Devuelve un dict {equipo: P(gana en penales)} con el suavizado aplicado.
    """
    df = pd.read_csv(path)

    home = df[["home_team", "winner"]].copy()
    home.columns = ["team", "winner"]
    away = df[["away_team", "winner"]].copy()
    away.columns = ["team", "winner"]
    all_matches = pd.concat([home, away], ignore_index=True)
    all_matches["won"] = all_matches["team"] == all_matches["winner"]

    stats = all_matches.groupby("team").agg(
        played=("won", "count"),
        won_count=("won", "sum")
    ).reset_index()

    # Suavizado bayesiano: P_suavizado = (victorias + prior * P_global) / (partidos + prior)
    # Con prior=10 y P_global=0.511, un equipo con 0/3 = 0.0% crudo → 18.6% suavizado
    # Un equipo con 23/25 = 92% crudo → 83.7% suavizado (más conservador con los outliers)
    stats["win_rate_smoothed"] = (
        (stats["won_count"] + SMOOTHING_PRIOR * GLOBAL_PENALTY_WIN_RATE) /
        (stats["played"] + SMOOTHING_PRIOR)
    )

    return dict(zip(stats["team"], stats["win_rate_smoothed"]))


def predict_penalty_winner(home_team, away_team,
                            home_shoots_first=True,
                            shootout_stats=None,
                            path="shootouts.csv"):
    """
    Predice la probabilidad de ganar en penales para cada equipo.

    Args:
        home_team, away_team: nombres exactos (igual que en results.csv)
        home_shoots_first: True si el equipo local patea primero.
                           Si no se sabe, pasar None para ignorar el ajuste.
        shootout_stats: dict pre-cargado (para no releer el CSV si hacés
                        múltiples predicciones). Si None, lo carga del CSV.
        path: ruta a shootouts.csv

    Returns:
        dict con:
          - 'p_home': P(gana en penales el equipo local)
          - 'p_away': P(gana en penales el equipo visitante)
          - 'home_penalty_rate': tasa histórica suavizada del local
          - 'away_penalty_rate': tasa histórica suavizada del visitante
          - 'home_played': partidos de penales en el historial del local
          - 'away_played': partidos de penales en el historial del visitante
          - 'first_shooter_advantage_applied': si se aplicó el ajuste o no
    """
    if shootout_stats is None:
        shootout_stats = load_shootout_stats(path)

    # Tasas base (suavizadas)
    p_home_base = shootout_stats.get(home_team, GLOBAL_PENALTY_WIN_RATE)
    p_away_base = shootout_stats.get(away_team, GLOBAL_PENALTY_WIN_RATE)

    # Normalizar para que sumen 1 (son complementarias)
    total = p_home_base + p_away_base
    p_home_norm = p_home_base / total
    p_away_norm = p_away_base / total

    # Ajuste por ventaja del primer pateador
    # La ventaja histórica es 53.1% vs 46.9%, es decir +3.1% vs la base 50/50.
    # Aplicamos ese delta de forma proporcional sobre las probabilidades ya normalizadas.
    first_shooter_delta = FIRST_SHOOTER_ADVANTAGE - 0.5  # = +0.031

    if home_shoots_first is True:
        p_home_final = p_home_norm + first_shooter_delta * (1 - p_home_norm)
        p_away_final = 1 - p_home_final
        applied = True
    elif home_shoots_first is False:
        p_away_final = p_away_norm + first_shooter_delta * (1 - p_away_norm)
        p_home_final = 1 - p_away_final
        applied = True
    else:
        # No se sabe quién patea primero, se omite el ajuste
        p_home_final = p_home_norm
        p_away_final = p_away_norm
        applied = False

    # Historial crudo para mostrar al usuario
    df = pd.read_csv(path)
    home_all = pd.concat([df[df.home_team == home_team], df[df.away_team == home_team]])
    away_all = pd.concat([df[df.home_team == away_team], df[df.away_team == away_team]])

    return {
        "p_home": round(p_home_final, 4),
        "p_away": round(p_away_final, 4),
        "home_penalty_rate": round(p_home_base, 4),
        "away_penalty_rate": round(p_away_base, 4),
        "home_played": len(home_all),
        "away_played": len(away_all),
        "first_shooter_advantage_applied": applied,
    }


def print_penalty_prediction(home_team, away_team,
                              home_shoots_first=None,
                              shootout_stats=None,
                              path="shootouts.csv"):
    """Helper para imprimir la predicción de penales de forma legible."""
    result = predict_penalty_winner(
        home_team, away_team, home_shoots_first, shootout_stats, path)

    print(f"\n{'='*50}")
    print(f"PENALES: {home_team} vs {away_team}")
    print(f"{'='*50}")
    print(f"  {home_team} gana: {result['p_home']:.1%}")
    print(f"    (historial: {result['home_played']} definiciones)")
    print(f"  {away_team} gana: {result['p_away']:.1%}")
    print(f"    (historial: {result['away_played']} definiciones)")
    if result["first_shooter_advantage_applied"]:
        shooter = home_team if home_shoots_first else away_team
        print(f"  Ajuste: +{(FIRST_SHOOTER_ADVANTAGE - 0.5):.1%} aplicado a {shooter} (patea primero)")
    else:
        print(f"  Sin ajuste de primer pateador (no especificado)")
    return result


if __name__ == "__main__":
    import sys

    # Cargar stats una sola vez
    stats = load_shootout_stats()
    print(f"Equipos con historial de penales: {len(stats)}")

    # Ejemplo: Argentina vs Francia (memorable)
    home = sys.argv[1] if len(sys.argv) > 1 else "Argentina"
    away = sys.argv[2] if len(sys.argv) > 2 else "France"

    print_penalty_prediction(home, away, home_shoots_first=True, shootout_stats=stats)
    print()

    # Mostrar los 10 mejores y 10 peores equipos en penales (con 5+ partidos)
    df_stats = pd.DataFrame([
        {"team": t, "rate": r, "played": len(pd.concat([
            pd.read_csv("shootouts.csv").query(f"home_team == '{t}' or away_team == '{t}'")]))}
        for t, r in stats.items()
    ])
    # Versión más simple sin el read_csv en el loop
    df_raw = pd.read_csv("shootouts.csv")
    home_c = df_raw[["home_team", "winner"]].rename(columns={"home_team": "team"})
    away_c = df_raw[["away_team", "winner"]].rename(columns={"away_team": "team"})
    played_counts = pd.concat([home_c, away_c]).groupby("team").size().reset_index(name="played")
    df_stats = pd.DataFrame(list(stats.items()), columns=["team", "smoothed_rate"])
    df_stats = df_stats.merge(played_counts, on="team")
    df_stats = df_stats[df_stats.played >= 5].sort_values("smoothed_rate", ascending=False)

    print("\nTOP 10 mejores en penales (≥5 definiciones):")
    print(df_stats.head(10)[["team", "played", "smoothed_rate"]].to_string(index=False))
    print("\nTOP 10 peores en penales (≥5 definiciones):")
    print(df_stats.tail(10)[["team", "played", "smoothed_rate"]].to_string(index=False))
