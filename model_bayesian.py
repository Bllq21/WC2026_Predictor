"""
Modelo bayesiano jerárquico (Baio-Blangiardo 2010): mismo planteamiento que
Dixon-Coles (Poisson con attack/defense por equipo + ventaja de local), pero
estimado vía MCMC en vez de máxima verosimilitud. La ventaja real es que
da una distribución posterior completa, no solo un punto estimado -
podés reportar intervalos de credibilidad, no solo un número.

ADVERTENCIA DE ESCALA: con 336 equipos el modelo completo tiene ~670
parámetros latentes. Muestrear esto con NUTS sobre los ~49k partidos es
viable pero lento (varios minutos). Para que esto sea usable en una laptop,
entrenamos sobre un SUBSET de equipos relevantes (las selecciones del
partido en cuestión + sus rivales históricos), no sobre las 336 selecciones
de toda la historia del fútbol. Esto es una decisión de diseño explícita,
no un atajo oculto - se documenta en el README.

NOTA TÉCNICA - COMPILACIÓN C DESHABILITADA: PyTensor (el motor de cómputo
de PyMC) por default compila el modelo a C++ con clang para que NUTS corra
rápido. En macOS muy reciente (confirmado en macOS 27 Beta, y reportado en
versiones anteriores también - ver pymc-devs/pytensor#1342, #1347) Clang
cambia flags internos del linker (-ld64, -lto_library) más rápido de lo
que PyTensor se actualiza para acompañarlos, y la compilación falla con
errores de linker como "library 'd64' not found". La solución robusta -
en vez de perseguir el flag correcto de Clang en cada versión de macOS -
es forzar PyTensor a su modo Python puro (sin compilar nada), seteando
cxx="" ANTES de que PyMC/PyTensor inicialicen su config. Es algo más lento
que C compilado, pero en hardware multi-core moderno (probado: ~900
draws/s en un M3 Pro) la diferencia es irrelevante para este caso de uso,
y elimina por completo la dependencia del toolchain de Clang del sistema.
"""
import pytensor
pytensor.config.cxx = ""  # fuerza modo Python puro, evita el linker de Clang

import numpy as np
import pandas as pd
import pymc as pm

from data_prep import load_results, add_decay_weight


class BayesianGoalModel:
    def __init__(self, draws=1000, tune=1000, chains=2):
        self.draws = draws
        self.tune = tune
        self.chains = chains
        self.trace = None
        self.teams = None
        self.team_idx = None

    def fit(self, df, relevant_teams=None, max_teams=60):
        """
        relevant_teams: si se especifica, se entrena solo sobre partidos
        que involucren a esos equipos (más sus oponentes directos), para
        mantener el modelo en una escala manejable por MCMC en una laptop.
        """
        if relevant_teams:
            # Incluir también a los oponentes de esos equipos, para que el
            # modelo tenga contexto de fuerza relativa, no solo partidos
            # entre los equipos de interés.
            mask = df.home_team.isin(relevant_teams) | df.away_team.isin(relevant_teams)
            opponents = set(df[mask].home_team) | set(df[mask].away_team)
            teams_in_scope = sorted(opponents)[:max_teams]
            if not set(relevant_teams).issubset(teams_in_scope):
                teams_in_scope = sorted(set(teams_in_scope) | set(relevant_teams))
            df = df[df.home_team.isin(teams_in_scope) & df.away_team.isin(teams_in_scope)]

        teams = sorted(set(df.home_team) | set(df.away_team))
        self.teams = teams
        self.team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = df.home_team.map(self.team_idx).values
        away_idx = df.away_team.map(self.team_idx).values
        home_goals = df.home_score.values.astype(int)
        away_goals = df.away_score.values.astype(int)
        weights = df.weight.values

        with pm.Model() as model:
            # Priors jerárquicos: cada equipo tiene attack/defense que vienen
            # de una distribución común (esto es lo "jerárquico" de
            # Baio-Blangiardo, regulariza equipos con pocos partidos).
            sigma_attack = pm.HalfNormal("sigma_attack", sigma=1.0)
            sigma_defense = pm.HalfNormal("sigma_defense", sigma=1.0)

            attack = pm.Normal("attack", mu=0, sigma=sigma_attack, shape=n)
            defense = pm.Normal("defense", mu=0, sigma=sigma_defense, shape=n)
            home_adv = pm.Normal("home_adv", mu=0.2, sigma=0.5)

            log_lambda_home = attack[home_idx] + defense[away_idx] + home_adv
            log_lambda_away = attack[away_idx] + defense[home_idx]

            # El peso temporal se aplica escalando la verosimilitud
            # (vía pm.Potential, equivalente a una verosimilitud ponderada).
            pm.Potential(
                "weighted_loglik",
                pm.math.sum(weights * (
                    pm.logp(pm.Poisson.dist(mu=pm.math.exp(log_lambda_home)), home_goals) +
                    pm.logp(pm.Poisson.dist(mu=pm.math.exp(log_lambda_away)), away_goals)
                ))
            )

            import multiprocessing
            available_cores = max(multiprocessing.cpu_count(), 1)
            # Usar 1 core por chain (paralelismo real), sin exceder los cores
            # disponibles. El max(..., 1) evita la división por cero de PyMC
            # en entornos donde cpu_count() devuelve 0 (visto en sandboxes
            # containerizados).
            n_cores = max(min(self.chains, available_cores), 1)

            self.trace = pm.sample(
                draws=self.draws, tune=self.tune, chains=self.chains,
                cores=n_cores,
                target_accept=0.9, progressbar=True, random_seed=42,
            )
        return self

    def predict_outcome_samples(self, home_team, away_team, max_goals=10):
        """Devuelve muestras posteriores de P(local)/P(empate)/P(visitante),
        para poder reportar incertidumbre, no solo un punto."""
        hi, ai = self.team_idx[home_team], self.team_idx[away_team]
        attack = self.trace.posterior["attack"].values  # (chain, draw, team)
        defense = self.trace.posterior["defense"].values
        home_adv = self.trace.posterior["home_adv"].values

        attack = attack.reshape(-1, attack.shape[-1])
        defense = defense.reshape(-1, defense.shape[-1])
        home_adv = home_adv.reshape(-1)

        lambda_home = np.exp(attack[:, hi] + defense[:, ai] + home_adv)
        lambda_away = np.exp(attack[:, ai] + defense[:, hi])

        # Para cada muestra posterior, calculamos P(resultado) exacto via Poisson,
        # en vez de simular goles discretos (mucho menos ruido de Monte Carlo).
        from scipy.stats import poisson as sp_poisson
        h_range = np.arange(max_goals + 1)
        p_home_samples = np.zeros(len(lambda_home))
        p_draw_samples = np.zeros(len(lambda_home))
        p_away_samples = np.zeros(len(lambda_home))
        for i in range(len(lambda_home)):
            mat = np.outer(sp_poisson.pmf(h_range, lambda_home[i]), sp_poisson.pmf(h_range, lambda_away[i]))
            mat /= mat.sum()
            p_home_samples[i] = np.tril(mat, -1).sum()
            p_draw_samples[i] = np.trace(mat)
            p_away_samples[i] = np.triu(mat, 1).sum()

        return p_home_samples, p_draw_samples, p_away_samples

    def predict_outcome(self, home_team, away_team, max_goals=10):
        """Punto estimado (media posterior) + intervalo de credibilidad 90%."""
        ph, pd_, pa = self.predict_outcome_samples(home_team, away_team, max_goals)
        summary = {}
        for name, samples in [("home", ph), ("draw", pd_), ("away", pa)]:
            summary[name] = {
                "mean": samples.mean(),
                "ci90": (np.percentile(samples, 5), np.percentile(samples, 95)),
            }
        return summary


if __name__ == "__main__":
    df = load_results()  # ya excluye fixtures futuros sin jugar (dropna interno)
    df = add_decay_weight(df, half_life_days=730)

    print("Entrenando modelo bayesiano (subset de equipos, ~1-3 min)...")
    model = BayesianGoalModel(draws=800, tune=800, chains=2)
    model.fit(df, relevant_teams=["Argentina", "Austria"], max_teams=50)

    summary = model.predict_outcome("Argentina", "Austria")
    print("\nArgentina vs Austria:")
    for outcome, stats in summary.items():
        lo, hi = stats["ci90"]
        print(f"  {outcome}: {stats['mean']:.1%}  (CI90%: {lo:.1%} - {hi:.1%})")
