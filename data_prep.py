"""
Carga y preparación de datos para el predictor de partidos.
Incluye time-decay weighting (estilo Dixon-Coles) para que partidos
recientes pesen mucho más que partidos históricos.
"""
import pandas as pd
import numpy as np

def load_results(path="results.csv", min_date=None, dropna_scores=True):
    """
    Carga el dataset completo. min_date es opcional (string 'YYYY-MM-DD').

    NOTA TÉCNICA - BUG DE PLATAFORMA: en cierta combinación de pandas 3.0.x
    + numpy 2.4.x en macOS ARM64, CUALQUIER operación no trivial sobre una
    columna datetime64 nativa de pandas (sort_values, comparaciones,
    aritmética de Timedelta) puede disparar un "bus error" real (crash de
    memoria de bajo nivel, sin excepción de Python, confirmado con
    faulthandler apuntando a pandas._libs.tslibs). No es un bug de este
    código: es una incompatibilidad binaria de esa combinación específica
    de versiones en esa arquitectura.

    Por eso, en vez de parchear cada operación una por una, esta función
    NUNCA expone una columna datetime64 nativa. La fecha se guarda como:
      - 'date_str': el string original 'YYYY-MM-DD' (para mostrar)
      - 'date_ordinal': entero = días desde 1970-01-01 (para ordenar,
        comparar, y calcular decay - aritmética de enteros, no de
        datetime64, así que no toca el código C problemático)
    Cualquier código que necesite "la fecha" para lógica (ordenar, decay,
    filtrar por rango) debe usar 'date_ordinal', no una columna datetime.
    """
    df = pd.read_csv(path)
    if dropna_scores:
        df = df.dropna(subset=["home_score", "away_score"])

    df = df.rename(columns={"date": "date_str"})
    # date.toordinal() vía datetime.date estándar de Python (NO pandas
    # datetime64) - aritmética/parsing 100% en Python puro, sin tocar
    # pandas._libs.tslibs en absoluto.
    import datetime as _dt
    df["date_ordinal"] = df["date_str"].apply(
        lambda s: _dt.date.fromisoformat(s).toordinal()
    )

    if min_date:
        min_ordinal = _dt.date.fromisoformat(min_date).toordinal()
        df = df[df["date_ordinal"] >= min_ordinal]

    return df.reset_index(drop=True)


def add_decay_weight(df, half_life_days=730, reference_date=None):
    """
    Agrega una columna 'weight' con decaimiento exponencial, usando
    'date_ordinal' (entero) en vez de aritmética datetime64 nativa de
    pandas - ver nota técnica en load_results() sobre por qué.

    reference_date, si se da, debe ser un string 'YYYY-MM-DD' (no un
    Timestamp de pandas).
    """
    import datetime as _dt
    if reference_date is None:
        reference_ordinal = df["date_ordinal"].max()
    elif isinstance(reference_date, str):
        reference_ordinal = _dt.date.fromisoformat(reference_date).toordinal()
    else:
        reference_ordinal = reference_date  # ya es un ordinal entero

    days_ago = (reference_ordinal - df["date_ordinal"]).clip(lower=0)
    decay_rate = np.log(2) / half_life_days
    df = df.copy()
    df["weight"] = np.exp(-decay_rate * days_ago)
    return df


# Pesos por tipo de torneo, basados en valores reales de la columna
# 'tournament' del dataset (confirmados con value_counts() sobre 49k partidos).
# Tier 1 = máxima competitividad; Tier 4 = amistosos/invitacionales.
# Valores diseñados para que la diferencia sea perceptible pero no radical:
# un Friendly reciente (decay alto) sigue valiendo más que un Mundial de 1980
# (decay casi cero), porque el decay temporal sigue dominando.
TOURNAMENT_WEIGHTS = {
    # Tier 1: torneos finales de máxima competitividad (peso 1.0)
    "FIFA World Cup": 1.0,
    "Copa América": 1.0,
    "UEFA Euro": 1.0,
    "African Cup of Nations": 1.0,
    "AFC Asian Cup": 1.0,
    "Gold Cup": 1.0,
    "CONCACAF Championship": 1.0,
    "Confederations Cup": 1.0,
    "Olympic Games": 1.0,
    "UEFA Nations League": 0.95,       # competitivo pero sin el estatus de torneo mayor
    "CONCACAF Nations League": 0.95,

    # Tier 2: eliminatorias de torneos Tier 1 (peso 0.85)
    # partidos con puntos reales en juego, selecciones completas obligadas
    "FIFA World Cup qualification": 0.85,
    "UEFA Euro qualification": 0.85,
    "African Cup of Nations qualification": 0.85,
    "AFC Asian Cup qualification": 0.85,
    "Gold Cup qualification": 0.85,
    "CONCACAF Championship qualification": 0.85,
    "Copa América qualification": 0.85,
    "Oceania Nations Cup qualification": 0.85,
    "CONCACAF Nations League qualification": 0.85,
    "EAFF Championship qualification": 0.80,
    "AFF Championship qualification": 0.80,

    # Tier 3: torneos regionales con competitividad real pero menor escala (peso 0.65)
    "AFC Challenge Cup": 0.65,
    "AFC Challenge Cup qualification": 0.65,
    "AFF Championship": 0.65,
    "CECAFA Cup": 0.65,
    "COSAFA Cup": 0.65,
    "Gulf Cup": 0.65,
    "CFU Caribbean Cup": 0.65,
    "CFU Caribbean Cup qualification": 0.65,
    "UNCAF Cup": 0.65,
    "WAFF Championship": 0.65,
    "SAFF Cup": 0.65,
    "EAFF Championship": 0.65,
    "Arab Cup": 0.65,
    "Arab Cup qualification": 0.65,
    "COSAFA Cup qualification": 0.65,

    # Tier 4: amistosos e invitacionales (peso 0.35)
    # El modelo los sigue viendo, solo con menos influencia sobre attack/defense
    "Friendly": 0.35,
    "King's Cup": 0.40,
    "Kirin Cup": 0.40,
    "Korea Cup": 0.40,
    "Merdeka Tournament": 0.40,
    "Cyprus International Tournament": 0.40,
    "Nehru Cup": 0.40,
}

# Peso default para cualquier torneo no listado explícitamente.
# Deliberadamente en Tier 3 (no Tier 4) porque la mayoría de los no-listados
# son torneos regionales menores con alguna competitividad real, no amistosos.
TOURNAMENT_WEIGHT_DEFAULT = 0.60


def add_tournament_weight(df, weights=None, default=None):
    """
    Multiplica la columna 'weight' existente (del decay temporal) por un
    factor de competitividad según el tipo de torneo.

    Debe llamarse DESPUÉS de add_decay_weight(), no antes.
    El weight final = decay_weight * tournament_weight, así que el decay
    temporal sigue dominando: un partido del Mundial 1980 (decay ~0.0)
    sigue valiendo casi cero aunque sea Tier 1.

    Args:
        df: DataFrame con columna 'weight' ya calculada por add_decay_weight()
        weights: dict custom de {tournament: factor}. Si None, usa TOURNAMENT_WEIGHTS
        default: factor para torneos no listados. Si None, usa TOURNAMENT_WEIGHT_DEFAULT
    """
    if weights is None:
        weights = TOURNAMENT_WEIGHTS
    if default is None:
        default = TOURNAMENT_WEIGHT_DEFAULT

    df = df.copy()
    tournament_factors = df["tournament"].map(weights).fillna(default)
    df["weight"] = df["weight"] * tournament_factors
    return df


def filter_teams(df, teams):
    """Filtra el dataset a partidos donde participó al menos uno de 'teams'.
    Útil para inspección rápida; los modelos usan el dataset completo."""
    mask = df["home_team"].isin(teams) | df["away_team"].isin(teams)
    return df[mask]


if __name__ == "__main__":
    df = load_results()
    print(f"Partidos totales: {len(df)}")
    print(f"Rango de fechas: {df['date_str'].min()} a {df['date_str'].max()}")

    df_w = add_decay_weight(df)
    df_w = add_tournament_weight(df_w)
    print(f"\nPeso de un partido de hoy: {df_w['weight'].max():.4f}")
    print(f"Peso de un Friendly reciente: {df_w[df_w['tournament']=='Friendly']['weight'].tail(5).mean():.4f}")
    print(f"Peso de un Mundial reciente:  {df_w[df_w['tournament']=='FIFA World Cup']['weight'].tail(5).mean():.4f}")
    print(f"\nDistribución de pesos por tier (media, partidos recientes):")
    recent = df_w[df_w["date_ordinal"] > df_w["date_ordinal"].max() - 730]
    print(recent.groupby("tournament")["weight"].mean().sort_values(ascending=False).head(10).to_string())
