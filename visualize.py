"""
Visualizaciones: bar chart de probabilidad de resultado, top-N marcadores
más probables, y heatmap completo de la matriz de marcadores.
Diseñadas para aceptar el output de CUALQUIERA de los 3 modelos, siempre
que devuelvan (matrix, p_home, p_draw, p_away) en el mismo formato.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_outcome_bars(home_team, away_team, p_home, p_draw, p_away,
                       model_name="", save_path=None):
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = [f"{home_team}\ngana", "Empate", f"{away_team}\ngana"]
    values = [p_home * 100, p_draw * 100, p_away * 100]
    colors = ["#6fa8dc", "#b0b0b0", "#e06666"]

    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", fontweight="bold")

    ax.set_ylabel("Probabilidad (%)")
    ax.set_ylim(0, max(values) * 1.2)
    title = f"Predicción: {home_team} vs {away_team}"
    if model_name:
        title += f"\nModelo: {model_name}"
    ax.set_title(title, fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_top_scorelines(home_team, away_team, top_n_list, save_path=None):
    """top_n_list: lista de (home_goals, away_goals, prob), ya ordenada."""
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [f"{home_team[:3].upper()} {h}-{a} {away_team[:3].upper()}" for h, a, p in top_n_list]
    values = [p * 100 for h, a, p in top_n_list]

    # Colorear por quién gana ese marcador (azul=local, gris=empate, rojo=visitante)
    colors = []
    for h, a, p in top_n_list:
        if h > a:
            colors.append("#6fa8dc")
        elif h == a:
            colors.append("#b0b0b0")
        else:
            colors.append("#e06666")

    y_pos = np.arange(len(labels))[::-1]
    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    for y, v in zip(y_pos, values):
        ax.text(v + 0.1, y, f"{v:.1f}%", va="center")

    ax.set_xlabel("Probabilidad (%)")
    ax.set_title(f"Top {len(top_n_list)} marcadores más probables\n{home_team} vs {away_team}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_scoreline_heatmap(home_team, away_team, matrix, max_goals=6, save_path=None):
    """Heatmap completo de la matriz de probabilidad de marcador."""
    fig, ax = plt.subplots(figsize=(7, 6))
    display_matrix = matrix[:max_goals + 1, :max_goals + 1] * 100

    sns.heatmap(
        display_matrix, annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=range(max_goals + 1), yticklabels=range(max_goals + 1),
        cbar_kws={"label": "Probabilidad (%)"}, ax=ax,
    )
    ax.set_xlabel(f"Goles {away_team}")
    ax.set_ylabel(f"Goles {home_team}")
    ax.set_title(f"Heatmap de marcadores: {home_team} vs {away_team}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_model_comparison(results_dict, save_path=None):
    """results_dict: {'Dixon-Coles': (acc, rps), 'XGBoost': (acc, rps), ...}"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    names = list(results_dict.keys())
    accs = [results_dict[n][0] * 100 for n in names]
    rpss = [results_dict[n][1] for n in names]
    colors = ["#808080", "#e06666", "#6fa8dc"][:len(names)]

    bars1 = axes[0].bar(names, accs, color=colors)
    for bar, v in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.3, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].set_title("Accuracy 1X2 (mayor = mejor)")
    axes[0].set_ylabel("%")

    bars2 = axes[1].bar(names, rpss, color=colors)
    for bar, v in zip(bars2, rpss):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold")
    axes[1].set_title("RPS (menor = mejor)")
    axes[1].set_ylabel("RPS")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_penalty_analysis(home_team, away_team, penalty_result, save_path=None):
    """
    Visualiza la prediccion de penales: probabilidades de victoria + contexto
    del historial de cada equipo.

    penalty_result: dict devuelto por model_penalties.predict_penalty_winner()
    """
    p_home = penalty_result["p_home"] * 100
    p_away = penalty_result["p_away"] * 100
    home_rate = penalty_result["home_penalty_rate"] * 100
    away_rate = penalty_result["away_penalty_rate"] * 100
    home_played = penalty_result["home_played"]
    away_played = penalty_result["away_played"]
    first_applied = penalty_result["first_shooter_advantage_applied"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel izquierdo: P(gana en penales)
    ax = axes[0]
    bars = ax.bar(
        [f"{home_team}\ngana", f"{away_team}\ngana"],
        [p_home, p_away],
        color=["#6fa8dc", "#e06666"],
        width=0.5,
    )
    for bar, v in zip(bars, [p_home, p_away]):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                f"{v:.1f}%", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylim(0, max(p_home, p_away) * 1.25)
    ax.set_ylabel("Probabilidad (%)")
    subtitle = "con ajuste de primer pateador" if first_applied else "sin ajuste de primer pateador"
    ax.set_title(f"Penales: {home_team} vs {away_team}\n({subtitle})",
                 fontsize=11, fontweight="bold")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1.45, 51, "50%", color="gray", fontsize=8)

    # Panel derecho: historial crudo con suavizado
    ax2 = axes[1]
    teams = [home_team, away_team]
    raw_rates = [home_rate, away_rate]
    played_counts = [home_played, away_played]
    colors_hist = ["#6fa8dc", "#e06666"]

    bars2 = ax2.bar(teams, raw_rates, color=colors_hist, width=0.4, alpha=0.85)
    for bar, v, n in zip(bars2, raw_rates, played_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1,
                 f"{v:.1f}%\n({n} partidos)", ha="center", fontsize=10)
    ax2.set_ylim(0, max(raw_rates) * 1.35)
    ax2.set_ylabel("Tasa historica suavizada (%)")
    ax2.set_title("Historial de penales\n(suavizado bayesiano, prior=10 partidos)",
                  fontsize=11, fontweight="bold")
    ax2.axhline(51.1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.text(1.25, 52, "Media global", color="gray", fontsize=8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_ensemble_prediction(home_team, away_team, model_results,
                              weights=None, save_path=None):
    """
    Grafica el promedio ponderado de los modelos disponibles, mostrando
    tanto el resultado de cada modelo como el ensemble final.

    model_results: dict {nombre_modelo: (p_home, p_draw, p_away)}
                   Puede incluir cualquier subconjunto de modelos.
    weights: dict {nombre_modelo: peso}. Si None, pesos iguales.

    El ensemble NO es solo un promedio simple: usa pesos que reflejan
    el accuracy historico de cada modelo (Dixon-Coles > Bayesian > XGBoost
    segun los benchmarks). Ajustable via el parametro weights.
    """
    if weights is None:
        # Pesos por defecto basados en accuracy del benchmark (evaluate.py 612 partidos)
        # Dixon-Coles: 60.5%, Bayesian: ~53% (15 partidos, referencial),
        # XGBoost: 56.2%, Elo: sin benchmark formal aun, peso conservador.
        default_weights = {
            "dixon-coles": 0.40,
            "bayesian": 0.25,
            "xgboost": 0.25,
            "elo": 0.10,
        }
        weights = {k: default_weights.get(k, 0.20) for k in model_results}

    # Normalizar pesos a los modelos disponibles
    total_w = sum(weights[m] for m in model_results if m in weights)
    norm_weights = {m: weights.get(m, 0.20) / total_w for m in model_results}

    # Ensemble ponderado
    ens_home = sum(model_results[m][0] * norm_weights[m] for m in model_results)
    ens_draw = sum(model_results[m][1] * norm_weights[m] for m in model_results)
    ens_away = sum(model_results[m][2] * norm_weights[m] for m in model_results)

    # Layout: un panel por modelo + uno para el ensemble
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models + 1, figsize=(4 * (n_models + 1), 5.5))

    model_colors = {
        "dixon-coles": "#6fa8dc",
        "xgboost": "#e06666",
        "bayesian": "#93c47d",
        "elo": "#f6b26b",
    }
    outcome_colors = ["#4a86c8", "#888888", "#c94444"]
    labels = [f"{home_team}\ngana", "Empate", f"{away_team}\ngana"]

    for i, (model_name, (ph, pd, pa)) in enumerate(model_results.items()):
        ax = axes[i]
        values = [ph * 100, pd * 100, pa * 100]
        color = model_colors.get(model_name, "#aaaaaa")
        bars = ax.bar(labels, values,
                      color=[c + "cc" for c in ["#4a86c8", "#888888", "#c94444"]],
                      edgecolor=color, linewidth=2.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", fontsize=9, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title(f"{model_name.upper()}\n(peso: {norm_weights[model_name]:.0%})",
                     fontsize=10, fontweight="bold", color=color)
        ax.set_ylabel("Probabilidad (%)" if i == 0 else "")
        ax.tick_params(axis="x", labelsize=8)

    # Ensemble
    ax_ens = axes[-1]
    ens_values = [ens_home * 100, ens_draw * 100, ens_away * 100]
    bars_ens = ax_ens.bar(labels, ens_values, color=outcome_colors, linewidth=0)
    for bar, v in zip(bars_ens, ens_values):
        ax_ens.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}%", ha="center", fontsize=11, fontweight="bold",
                    color="white" if v > 30 else "black")
    ax_ens.set_ylim(0, max(ens_values) * 1.25)
    ax_ens.set_title("ENSEMBLE\n(promedio ponderado)",
                     fontsize=11, fontweight="bold", color="#333333")
    ax_ens.spines["top"].set_visible(False)
    ax_ens.spines["right"].set_visible(False)
    for spine in ["left", "bottom"]:
        ax_ens.spines[spine].set_linewidth(2)

    fig.suptitle(f"{home_team} vs {away_team}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig, (ens_home, ens_draw, ens_away)

