"""
Punto de entrada único para predecir un partido. Cambiar de equipos NO
requiere tocar el código de los modelos - solo los argumentos acá.

Uso:
    python predict.py "Argentina" "Brasil"
    python predict.py "Argentina" "Austria" --model xgboost
    python predict.py "Argentina" "Austria" --model bayesian --neutral
    python predict.py "Argentina" "Austria" --model all --save
"""
import argparse
import sys

from data_prep import load_results, add_decay_weight, add_tournament_weight
from model_dixon_coles import DixonColes
from model_xgboost import build_team_features, XGBoostGoalModel
from model_bayesian import BayesianGoalModel
from model_penalties import print_penalty_prediction, load_shootout_stats, predict_penalty_winner
from model_elo import compute_elo, predict_elo, get_team_elo
from visualize import (plot_outcome_bars, plot_top_scorelines, plot_scoreline_heatmap,
                        plot_penalty_analysis, plot_ensemble_prediction)


def run_dixon_coles(df, home, away, args):
    df_w = add_decay_weight(df, half_life_days=args.half_life)
    if args.tournament_weights:
        df_w = add_tournament_weight(df_w)
    model = DixonColes().fit(df_w)
    if home not in model.team_idx or away not in model.team_idx:
        missing = [t for t in (home, away) if t not in model.team_idx]
        print(f"ERROR: equipo(s) no encontrado(s) en el dataset: {missing}")
        print("Revisá el nombre exacto en results.csv (case-sensitive).")
        sys.exit(1)
    p_home, p_draw, p_away = model.predict_outcome(home, away)
    top = model.top_scorelines(home, away, n=args.top_n)
    matrix = model.predict_scoreline_matrix(home, away)
    return p_home, p_draw, p_away, top, matrix


def run_xgboost(df, home, away, args):
    feat_df = build_team_features(df)
    model = XGBoostGoalModel().fit(feat_df)
    known = set(feat_df.home_team) | set(feat_df.away_team)
    missing = [t for t in (home, away) if t not in known]
    if missing:
        print(f"ERROR: equipo(s) no encontrado(s) en el dataset: {missing}")
        sys.exit(1)
    p_home, p_draw, p_away = model.predict_outcome(feat_df, home, away, neutral=int(args.neutral))
    top = model.top_scorelines(feat_df, home, away, n=args.top_n, neutral=int(args.neutral))
    matrix = model.predict_scoreline_matrix(feat_df, home, away, neutral=int(args.neutral))
    return p_home, p_draw, p_away, top, matrix


def run_bayesian(df, home, away, args):
    df_w = add_decay_weight(df, half_life_days=args.half_life)
    if args.tournament_weights:
        df_w = add_tournament_weight(df_w)
    model = BayesianGoalModel(draws=args.draws, tune=args.draws, chains=args.chains)
    model.fit(df_w, relevant_teams=[home, away], max_teams=args.max_teams)
    summary = model.predict_outcome(home, away)
    p_home, p_draw, p_away = summary["home"]["mean"], summary["draw"]["mean"], summary["away"]["mean"]
    print(f"  Intervalos de credibilidad 90%:")
    for k, v in summary.items():
        lo, hi = v["ci90"]
        print(f"    {k}: {lo:.1%} - {hi:.1%}")
    return p_home, p_draw, p_away, None, None  # bayesiano no expone matrix/top en esta versión


def run_elo(df, home, away, args):
    ratings, _ = compute_elo(df)
    if home not in ratings or away not in ratings:
        missing = [t for t in (home, away) if t not in ratings]
        print(f"ERROR: equipo(s) sin historial en el dataset: {missing}")
        import sys; sys.exit(1)
    neutral = True  # Elo siempre asume neutral para WC; ajustable si se agrega flag
    p_home, p_draw, p_away = predict_elo(ratings, home, away, neutral=neutral)
    print(f"  Elo {home}: {get_team_elo(ratings, home):.0f}")
    print(f"  Elo {away}: {get_team_elo(ratings, away):.0f}")
    print(f"  Diferencia: {get_team_elo(ratings, home) - get_team_elo(ratings, away):+.0f}")
    return p_home, p_draw, p_away, None, None  # Elo no genera heatmap ni top scorelines


MODELS = {
    "dixon-coles": run_dixon_coles,
    "xgboost": run_xgboost,
    "bayesian": run_bayesian,
    "elo": run_elo,
}


def main():
    parser = argparse.ArgumentParser(description="Predictor de partidos de fútbol internacional")
    parser.add_argument("home_team", help="Nombre exacto del equipo local (ver results.csv)")
    parser.add_argument("away_team", help="Nombre exacto del equipo visitante")
    parser.add_argument("--model", choices=["dixon-coles", "xgboost", "bayesian", "elo", "all"],
                         default="dixon-coles")
    parser.add_argument("--neutral", action="store_true", help="Cancha neutral (solo afecta XGBoost)")
    parser.add_argument("--tournament-weights", action="store_true",
                        help="Aplicar pesos por tipo de torneo encima del decay temporal "
                             "(Mundiales pesan más que Friendlies). Recomendado para mejorar "
                             "la precisión sin cambiar la metodología base.")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--half-life", type=int, default=730, help="Días para decay a la mitad")
    parser.add_argument("--draws", type=int, default=1500, help="Solo para --model bayesian")
    parser.add_argument("--chains", type=int, default=4, help="Solo para --model bayesian")
    parser.add_argument("--max-teams", type=int, default=100, help="Solo para --model bayesian")
    parser.add_argument("--save", action="store_true", help="Guardar graficas en outputs/")
    parser.add_argument("--penalties", action="store_true",
                        help="Prediccion de penales (eliminacion directa). No modifica modelos de 90.")
    parser.add_argument("--ensemble", action="store_true",
                        help="Graficar el promedio ponderado de todos los modelos corridos.")
    parser.add_argument("--first-shooter", choices=["home", "away", "unknown"],
                        default="unknown", help="Quien patea primero en penales.")
    args = parser.parse_args()

    df = load_results().sort_values("date_ordinal").reset_index(drop=True)

    models_to_run = list(MODELS.keys()) if args.model == "all" else [args.model]
    model_results_for_ensemble = {}  # acumula (p_home, p_draw, p_away) por modelo

    for model_name in models_to_run:
        print(f"\n{'='*50}\n{model_name.upper()}\n{'='*50}")
        p_home, p_draw, p_away, top, matrix = MODELS[model_name](df, args.home_team, args.away_team, args)

        print(f"{args.home_team} gana: {p_home:.1%}")
        print(f"Empate:          {p_draw:.1%}")
        print(f"{args.away_team} gana: {p_away:.1%}")
        model_results_for_ensemble[model_name] = (p_home, p_draw, p_away)

        if top:
            print(f"\nTop {len(top)} marcadores:")
            for h, a, p in top:
                print(f"  {h}-{a}: {p:.1%}")

        if args.save:
            slug = f"{args.home_team}_vs_{args.away_team}_{model_name}".replace(" ", "_")
            plot_outcome_bars(args.home_team, args.away_team, p_home, p_draw, p_away,
                               model_name=model_name, save_path=f"outputs/{slug}_bars.png")
            if top:
                plot_top_scorelines(args.home_team, args.away_team, top,
                                     save_path=f"outputs/{slug}_top_scores.png")
            if matrix is not None:
                plot_scoreline_heatmap(args.home_team, args.away_team, matrix,
                                        save_path=f"outputs/{slug}_heatmap.png")
            print(f"\nGráficas guardadas en outputs/{slug}_*.png")


    # Ensemble: grafico combinado de todos los modelos corridos
    if args.ensemble and len(model_results_for_ensemble) > 1:
        print(f"\n{'='*50}\nENSEMBLE\n{'='*50}")
        slug_ens = f"{args.home_team}_vs_{args.away_team}_ensemble".replace(" ", "_")
        save_path_ens = f"outputs/{slug_ens}.png" if args.save else None
        _, (ens_h, ens_d, ens_a) = plot_ensemble_prediction(
            args.home_team, args.away_team,
            model_results_for_ensemble,
            save_path=save_path_ens,
        )
        print(f"{args.home_team} gana: {ens_h:.1%}")
        print(f"Empate:          {ens_d:.1%}")
        print(f"{args.away_team} gana: {ens_a:.1%}")
        if args.save:
            print(f"Grafica ensemble guardada en outputs/{slug_ens}.png")
    elif args.ensemble:
        print("\n[ENSEMBLE] Se necesitan al menos 2 modelos. Usa --model all o combina modelos.")

    # Penales: capa separada
    if args.penalties:
        home_first = (True if args.first_shooter == "home"
                      else False if args.first_shooter == "away" else None)
        try:
            stats = load_shootout_stats()
            print_penalty_prediction(args.home_team, args.away_team,
                home_shoots_first=home_first, shootout_stats=stats)
            if args.save:
                pen_result = predict_penalty_winner(
                    args.home_team, args.away_team,
                    home_shoots_first=home_first,
                    shootout_stats=stats,
                )
                slug_pen = f"{args.home_team}_vs_{args.away_team}_penalties".replace(" ", "_")
                plot_penalty_analysis(
                    args.home_team, args.away_team, pen_result,
                    save_path=f"outputs/{slug_pen}.png",
                )
                print(f"Grafica de penales guardada en outputs/{slug_pen}.png")
        except FileNotFoundError:
            print("[PENALES] No se encontro shootouts.csv.")


if __name__ == "__main__":
    main()
