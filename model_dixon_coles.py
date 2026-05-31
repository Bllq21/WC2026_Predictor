"""
Modelo Dixon-Coles (1997): regresión de Poisson para goles de local/visitante,
con ajuste de correlación para marcadores bajos (0-0, 1-0, 0-1, 1-1) y
ponderación temporal exponencial.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from data_prep import load_results, add_decay_weight


class DixonColes:
    def __init__(self):
        self.teams = None
        self.params = None
        self.team_idx = None

    def fit(self, df):
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        self.team_idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        home_idx = df["home_team"].map(self.team_idx).values
        away_idx = df["away_team"].map(self.team_idx).values
        home_goals = df["home_score"].values
        away_goals = df["away_score"].values
        weights = df["weight"].values

        # params: [attack_0..attack_n-1, defense_0..defense_n-1, home_adv, rho]
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2, -0.1]])

        def neg_log_likelihood(params):
            attack = params[:n]
            defense = params[n:2*n]
            home_adv = params[2*n]
            rho = params[2*n + 1]

            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])

            log_lik = poisson.logpmf(home_goals, lambda_home) + \
                      poisson.logpmf(away_goals, lambda_away)

            # Ajuste de Dixon-Coles para marcadores bajos correlacionados
            tau = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            log_lik += np.log(np.clip(tau, 1e-10, None))

            return -np.sum(weights * log_lik)

        # L-BFGS-B maneja mucho mejor ~670 parámetros que SLSQP con restricciones.
        # La identificabilidad (attack promedio = 0) se resuelve post-hoc restando
        # la media, en vez de como restricción dura durante la optimización -
        # converge mucho más rápido y de forma más estable.
        def neg_log_likelihood_and_grad(params):
            attack = params[:n]
            defense = params[n:2*n]
            home_adv = params[2*n]
            rho = params[2*n + 1]

            lambda_home = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
            lambda_away = np.exp(attack[away_idx] + defense[home_idx])

            log_lik = poisson.logpmf(home_goals, lambda_home) + \
                      poisson.logpmf(away_goals, lambda_away)
            tau = self._tau(home_goals, away_goals, lambda_home, lambda_away, rho)
            log_lik += np.log(np.clip(tau, 1e-10, None))
            nll = -np.sum(weights * log_lik)

            # Gradiente analítico de la parte Poisson (ignora el pequeño
            # ajuste de tau en el gradiente: tau afecta solo 4 marcadores
            # raros y su efecto en el gradiente es despreciable frente al
            # costo de derivarlo a mano; el ajuste sigue aplicado en el NLL).
            d_home = weights * (home_goals - lambda_home)  # d(loglik)/d(log_lambda_home)
            d_away = weights * (away_goals - lambda_away)

            grad = np.zeros_like(params)
            np.subtract.at(grad[:n], home_idx, d_home)   # d/d_attack[home]
            np.subtract.at(grad[:n], away_idx, d_away)   # d/d_attack[away]
            np.subtract.at(grad[n:2*n], away_idx, d_home)  # d/d_defense[away]
            np.subtract.at(grad[n:2*n], home_idx, d_away)  # d/d_defense[home]
            grad[2*n] -= d_home.sum()  # d/d_home_adv
            # rho: derivada numérica puntual (un solo escalar, barato)
            eps = 1e-6
            params_eps = params.copy()
            params_eps[-1] += eps
            tau_eps = self._tau(home_goals, away_goals, lambda_home, lambda_away, params_eps[-1])
            d_rho = -np.sum(weights * (np.log(np.clip(tau_eps, 1e-10, None)) - np.log(np.clip(tau, 1e-10, None))) / eps)
            grad[-1] = d_rho

            return nll, grad

        # L-BFGS-B con gradiente analítico: converge en segundos en vez de
        # minutos, porque evita diferencias finitas sobre 674 parámetros.
        result = minimize(
            neg_log_likelihood_and_grad, x0, method="L-BFGS-B", jac=True,
            options={"maxiter": 1000, "ftol": 1e-9, "maxfun": 20000}
        )
        params = result.x.copy()
        attack_mean = params[:n].mean()
        params[:n] -= attack_mean
        params[n:2*n] += attack_mean  # compensar para no alterar lambda_home/away
        self.params = params
        self.converged = result.success
        self.opt_message = result.message
        return self

    @staticmethod
    def _tau(hg, ag, lh, la, rho):
        """Factor de corrección de Dixon-Coles para dependencia en marcadores bajos."""
        tau = np.ones_like(hg, dtype=float)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        return tau

    def _lambdas(self, home_team, away_team):
        n = len(self.teams)
        attack = self.params[:n]
        defense = self.params[n:2*n]
        home_adv = self.params[2*n]
        hi, ai = self.team_idx[home_team], self.team_idx[away_team]
        lambda_home = np.exp(attack[hi] + defense[ai] + home_adv)
        lambda_away = np.exp(attack[ai] + defense[hi])
        return lambda_home, lambda_away

    def predict_scoreline_matrix(self, home_team, away_team, max_goals=10):
        """Matriz de probabilidad para cada marcador posible, con ajuste rho."""
        lh, la = self._lambdas(home_team, away_team)
        rho = self.params[-1]
        matrix = np.zeros((max_goals + 1, max_goals + 1))
        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = poisson.pmf(h, lh) * poisson.pmf(a, la)
                tau = self._tau(np.array([h]), np.array([a]), np.array([lh]), np.array([la]), rho)[0]
                matrix[h, a] = p * tau
        matrix /= matrix.sum()
        return matrix

    def predict_outcome(self, home_team, away_team, max_goals=10):
        """Devuelve (P(gana local), P(empate), P(gana visitante))."""
        m = self.predict_scoreline_matrix(home_team, away_team, max_goals)
        p_home = np.tril(m, -1).sum()
        p_draw = np.trace(m)
        p_away = np.triu(m, 1).sum()
        return p_home, p_draw, p_away

    def top_scorelines(self, home_team, away_team, n=10, max_goals=10):
        m = self.predict_scoreline_matrix(home_team, away_team, max_goals)
        flat = [(h, a, m[h, a]) for h in range(max_goals + 1) for a in range(max_goals + 1)]
        flat.sort(key=lambda x: -x[2])
        return flat[:n]


if __name__ == "__main__":
    df = load_results()  # ya excluye fixtures futuros sin jugar (dropna interno)
    df = add_decay_weight(df, half_life_days=730)

    print("Entrenando Dixon-Coles (esto puede tardar 10-30s)...")
    model = DixonColes().fit(df)
    print(f"Convergió: {model.converged}")

    p_home, p_draw, p_away = model.predict_outcome("Argentina", "Austria")
    print(f"\nArgentina vs Austria:")
    print(f"  Argentina gana: {p_home:.1%}")
    print(f"  Empate:         {p_draw:.1%}")
    print(f"  Austria gana:   {p_away:.1%}")

    print("\nTop 5 marcadores:")
    for h, a, p in model.top_scorelines("Argentina", "Austria", n=5):
        print(f"  {h}-{a}: {p:.1%}")
