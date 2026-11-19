[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hqnsyhrn/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast — v2

Forecast, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at 2026-11-26T00:00:00Z, as five quantiles per
trophy. Entry for
[make.fun's gta6-trophy-rarity-forecast-01](https://make.fun/c/gta6-trophy-rarity-forecast-01).

## Model

An Exophase earn rate is a ratio — earners over players-tracked — and
the two sides move for different reasons, so the entry models them
separately:

- **Players-tracked N(t)** follows a log-logistic S-curve towards a
  ceiling nobody can see yet. The ceiling is unknown, so the entry
  runs an ensemble of ceilings, weighted by fit plus a wide prior, and
  forecasts the mixture — when the history cannot tell a world that
  saturates at 250k from one that runs to 600k, the median sits
  between them and the spread covers both.
- **Earners E(t)** follow, per trophy, the ODE
  `dE/dt = p·dN/dt + h·(N − E)`: new tracked players arrive already
  holding the trophy with probability `p`, and already-tracked
  non-earners earn at hazard `h` per hour. `p` and `h` are fitted per
  trophy against the observed N path, shrunk towards rate-dependent
  priors, on a fine grid. Earner growth is capped at 3× the fastest
  hourly gain the history ever saw, so no fit can extrapolate to a
  flood the data never showed.
- **Spread**: the upper tail is thinned in saturating scenarios (no new
  players arrive to lift rates), and a rolling-origin check on the
  case's own history calibrates the overall width.

`python3 main.py` reads every `*.in` file in `$MAKEFUN_INPUT_DIR`
(or `/input`) and writes one `<id>.out` per case to
`$MAKEFUN_OUTPUT_DIR` (or `/output`). Standard library only, no
network, deterministic, well under the 8 s limit.

## Validation (local, honest numbers)

Validated against practice cases, a live backtest (checkpoint 04:30Z
→ answer 11:12Z from the real Exophase page), and two synthetic
f00-shaped worlds with a truth mechanism different from the model's
own (see `../gta6-trophy-v2-work/validate.py` in the workspace — not
part of this repo). Pinball loss on the logit scale, lower is better:

| check | v2 | persistence |
|---|---:|---:|
| practice p00 (answer reconstructed, approx.) | 0.51 | 1.01 |
| practice p01 (answer reconstructed, approx.) | 0.47 | 1.01 |
| live backtest 04:30Z → 11:12Z (real Exophase page) | 0.16 | 0.58 |
| synthetic world A (N → 600k) | 0.40 | 0.70 |
| synthetic world B (N → 250k, earners 2× slower) | 0.26 | 0.33 |
| rolling origin, world A, 1-day horizon | 0.37 | 0.19 |
| rolling origin, world B, 1-day horizon | 0.29 | 0.49 |

All runs deterministic (two runs byte-identical) and inside the 8 s
limit (worst case 2.3 s on a full 2-day history).

The practice cases are near-unforecastable by construction (every read
before the checkpoint is the same frozen page; the answer is the first
refresh), yet the entry beats persistence there too, because the
N-ensemble expects the unobserved growth behind a stale page. The live
backtest and both synthetic worlds are the informative checks: the
model beats persistence clearly in world A and in the live backtest,
and now also in world B — the world v1 lost (0.46 vs 0.24). One known
limitation: at a 1-day horizon in world A the median moves too much
(0.37 vs persistence 0.19); the final's horizon is 4.5 days, where
the model wins clearly. Two variants were tried in parallel and
rejected on the live backtest: a maturity lag for the hazard pool
(live 0.44 vs 0.16) and 24 h half-life decay weighting of reads
(no help on the 1-day check, worse on world B).
