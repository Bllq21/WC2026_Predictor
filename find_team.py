"""
Helper para encontrar el nombre EXACTO de un equipo en el dataset.
Los nombres están en inglés y sin tildes (ej. "Mexico", no "México";
"Brazil", no "Brasil"). Usar esto antes de predict.py si no estás seguro.

Uso:
    python find_team.py argentina
    python find_team.py bras
"""
import sys
from data_prep import load_results


def find_team(query):
    df = load_results()
    teams = sorted(set(df.home_team) | set(df.away_team))
    query_lower = query.lower()
    matches = [t for t in teams if query_lower in t.lower()]
    return matches


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python find_team.py <texto a buscar>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])
    matches = find_team(query)
    if not matches:
        print(f"Sin coincidencias para '{query}'.")
    else:
        print(f"Coincidencias para '{query}':")
        for m in matches:
            print(f"  {m}")
