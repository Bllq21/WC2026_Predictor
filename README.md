# WC2026 Match Predictor

Pipeline de predicción de partidos de fútbol internacional. Compara varias
familias de modelos sobre el mismo dataset histórico y permite combinarlos en
un ensemble, además de estimar por separado la definición por penales.

Dataset: [international_results](https://github.com/martj42/international_results)
(martj42), 49k+ partidos internacionales desde 1872, CC0.

## Modelos

| Modelo | Enfoque | Salida |
| --- | --- | --- |
| **Dixon-Coles** | Poisson con corrección para marcadores bajos, estimado por MLE (`scipy.optimize`) | 1X2, top de marcadores y heatmap |
| **Bayesiano** | Poisson jerárquico estimado con MCMC (PyMC/NUTS) | 1X2 con intervalos de credibilidad |
| **XGBoost** | Gradient boosting con objetivo Poisson sobre features de ataque/defensa | 1X2, top de marcadores y heatmap |
| **Elo** | Ratings dinámicos actualizados partido a partido | 1X2 y diferencia de rating |
| **Penales** | Modelo histórico de shootouts | Probabilidad de ganar la tanda |
| **Ensemble** | Promedio de los modelos corridos en la misma ejecución | 1X2 combinado |

## Por qué varios modelos y no solo uno

Cada modelo representa una familia distinta de la literatura de predicción
deportiva: paramétrico clásico (Dixon-Coles), paramétrico bayesiano
(Baio-Blangiardo), machine learning no paramétrico (XGBoost), ratings
dinámicos (Elo) y una capa específica para eliminación directa (penales).
Compararlos sobre el mismo holdout permite hablar de trade-offs reales — no
solo "qué modelo es mejor" sino "mejor bajo qué métrica, a qué costo
computacional".

## Setup (macOS, con mise)

El repo incluye `mise.toml` con Python 3.12 y un `.envrc` para direnv.

```bash
# 1. Clonar y entrar al proyecto
git clone https://github.com/Bllq21/WC2026_Predictor.git
cd WC2026_Predictor

# 2. Confiar en la configuración de mise e instalar Python
mise trust
mise install

# 3. Crear el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 4. Instalar dependencias
pip install -r requirements.txt

# 5. Descargar los datasets (no están en el repo por tamaño/actualidad)
curl -sL -o results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv
curl -sL -o shootouts.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/shootouts.csv
```

Si usás direnv, ejecutá `direnv allow` una vez. El `.envrc` activa `.venv`
automáticamente al entrar a la carpeta.

### Nota sobre PyMC en Apple Silicon

PyMC usa PyTensor por debajo, que compila funciones en C. La primera vez que
corras `model_bayesian.py` puede tardar más por la compilación inicial (luego
queda en caché). Si aparecen errores de `gcc`/`clang`, instalá las Command
Line Tools de Xcode con `xcode-select --install`.

## Uso

```bash
# Buscar el nombre exacto de un equipo (nombres en inglés, sin tildes)
python find_team.py argentina

# Predecir un partido con Dixon-Coles (default)
python predict.py "Argentina" "Brazil"

# Elegir modelo
python predict.py "Argentina" "Austria" --model xgboost --neutral
python predict.py "Argentina" "Austria" --model bayesian --chains 4 --draws 1500
python predict.py "Argentina" "Austria" --model elo

# Correr todos los modelos, guardar gráficas y combinar el ensemble
python predict.py "Argentina" "Austria" --model all --ensemble --save

# Agregar la predicción de penales (requiere shootouts.csv)
python predict.py "Argentina" "Austria" --model all --ensemble --penalties --save

# Aplicar pesos por torneo además del decay temporal
python predict.py "Argentina" "Austria" --model all --tournament-weights --ensemble

# Evaluar accuracy/RPS de Dixon-Coles y XGBoost sobre los últimos N partidos
python evaluate.py 600

# Evaluar el modelo bayesiano (más lento: re-entrena MCMC por partido)
python evaluate_bayesian.py 20
```

**Importante:** los nombres de equipo están en inglés y sin tildes
(`Brazil`, no `Brasil`; `Mexico`, no `México`). Usá `find_team.py` si no
estás seguro del nombre exacto.

## Ejemplos visuales

Predicción combinada para Argentina vs Egipto:

![Ensemble Argentina vs Egipto](docs/ensemble.png)

Mapa de calor de marcadores (Dixon-Coles):

![Heatmap Argentina vs Egipto](docs/heatmap.png)

Análisis de la tanda de penales:

![Penales Argentina vs Egipto](docs/penalties.png)

Las imágenes se generan localmente con `--save` y se guardan en `outputs/`,
que está ignorado por git.

## Decisiones de diseño (y sus trade-offs)

- **Time-decay weighting (`half_life_days=730` por default):** se usa el
  dataset completo desde 1872, pero los partidos viejos pesan casi cero en el
  entrenamiento. Sin esto, mezclar fútbol de 1930 con fútbol actual distorsiona
  la fuerza real de los equipos. Ajustable vía `--half-life`.

- **Pesos por torneo opcionales (`--tournament-weights`):** los Mundiales y
  torneos continentales pesan más que los amistosos. Se aplican encima del
  decay temporal, no en lugar de él.

- **Modelo bayesiano limitado a un subconjunto de equipos por partido
  (`max_teams`, default 100):** NUTS con cientos de parámetros latentes sobre
  49k partidos es lento incluso en hardware potente. Se acota el universo a los
  equipos relevantes y sus rivales directos.

- **XGBoost sin tuning fino:** el modelo corre correctamente (sin data
  leakage, objetivo Poisson, features vectorizadas), pero los hiperparámetros
  son un punto de partida razonable, no el resultado de un grid search.

- **Elo se evalúa como modelo independiente:** usa ratings actualizados
  partido a partido y asume cancha neutral, consistente con un Mundial.

- **Penales como capa separada:** la tanda no se mezcla con la predicción de
  los 90 minutos. `predict.py --penalties` usa el historial de shootouts para
  estimar al ganador.

- **El benchmark accuracy/RPS excluye al modelo bayesiano del script
  principal** (`evaluate.py`) porque evaluarlo sobre un holdout grande
  significaría re-entrenar MCMC partido por partido. Se evalúa aparte, sobre
  un N más chico, en `evaluate_bayesian.py`.

## Estructura

```text
data_prep.py          # carga del CSV, decay temporal y pesos por torneo
model_dixon_coles.py  # Poisson + MLE (scipy.optimize, gradiente analítico)
model_xgboost.py      # features vectorizadas + XGBoost Poisson
model_bayesian.py     # Poisson jerárquico + PyMC/NUTS
model_elo.py          # ratings Elo y predicción 1X2
model_penalties.py    # historial de shootouts y probabilidad de la tanda
visualize.py          # barras, heatmap, ensemble y análisis de penales
predict.py            # CLI central: partido, modelo, ensemble y penales
find_team.py          # buscador de nombres exactos de equipo
evaluate.py           # benchmark accuracy/RPS (Dixon-Coles, XGBoost)
evaluate_bayesian.py  # benchmark accuracy/RPS (bayesiano, N chico)
docs/                 # gráficas de ejemplo
outputs/              # gráficas generadas con --save (ignorado por git)
```

## Métricas

- **Accuracy 1X2:** % de partidos donde el resultado más probable predicho
  (gana local / empate / gana visitante) coincidió con el real.
- **RPS (Rank Probability Score):** métrica más informativa que accuracy para
  outcomes ordinales — penaliza menos predecir "empate" cuando ganó el local
  que predecir "ganó el visitante" cuando en realidad ganó el local. Menor =
  mejor.

## Próximos pasos

- Tuning de hiperparámetros de XGBoost (grid search u Optuna).
- Calibración de probabilidades (Platt scaling / isotonic regression).
- Incorporar Elo como feature adicional de XGBoost.
- Subir `max_teams` del modelo bayesiano aprovechando más cores.
- Notebook exploratorio una vez estabilizado el código base.

## Licencia

MIT — ver [LICENSE](LICENSE).
