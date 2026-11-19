#!/usr/bin/env python3
"""GTA VI trophy rarity forecast, v2 (recorded rebuild).

Idea, in one paragraph: an Exophase earn rate is a ratio, earners over
players-tracked, and the two sides of that ratio move for different
reasons.  New players join (players-tracked grows, roughly an S-curve
towards a ceiling nobody can see yet) and some of them already hold a
trophy; players who are already tracked keep earning at a slow hazard.
So per trophy I fit the little ODE

    dE/dt = p * dN/dt + h * (N - E)

where E is earners, N is players-tracked, p is the share of brand-new
tracked players who already hold the trophy and h is the per-hour
hazard for the rest.  p and h are fitted per trophy, shrunk towards
rate-dependent priors, under an ensemble of N-curves (a log-logistic
S-curve whose ceiling is unknown: several ceilings, weighted by fit
plus a wide prior).  The forecast is the mixture over that ensemble,
so when the history cannot tell a fast-saturating world from a slow
one, the median sits between them and the spread covers both -- with
the upper tail thinned in saturating scenarios, where no new players
arrive to push rates up.  A rolling-origin check on the case's own
history calibrates the spread.  No network, no randomness, stdlib only.
"""

import bisect
import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
ZVALS = (-1.6448536269514722, -0.6744897501960817, 0.0,
         0.6744897501960817, 1.6448536269514722)
# Time zero, 12 h before the first read: players were already being
# tracked before release, so the N-curve's origin sits before the
# first read instead of at release midnight.
TIME_ZERO = datetime(2026, 11, 18, 12, tzinfo=timezone.utc)
EPS_RATE = 0.01      # rates are printed to 2 decimals; clamp into (0, 100)
MAX_RATE = 99.99


def parse_time(text):
    """ISO-8601 with Z or offset -> aware datetime (UTC)."""
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    return datetime.fromisoformat(text).astimezone(timezone.utc)


def hours_since_release(moment):
    """Hours from the time zero; floored so early reads stay positive."""
    return max((moment - TIME_ZERO).total_seconds() / 3600.0, 0.05)


def logit(pct):
    """logit of a percent in (0, 100)."""
    p = min(max(pct, EPS_RATE), MAX_RATE) / 100.0
    return math.log(p / (1.0 - p))


def expit(x):
    """Inverse logit -> percent in (0, 100)."""
    if x >= 0:
        p = 1.0 / (1.0 + math.exp(-x)) if x < 700 else 1.0
    else:
        e = math.exp(x)
        p = e / (1.0 + e)
    return min(max(p * 100.0, EPS_RATE), MAX_RATE)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def input_dir():
    return os.environ.get("MAKEFUN_INPUT_DIR") or "/input"


def output_dir():
    return os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"


def load_cases(indir):
    """Every *.in file in name order, as (case_id, case dict)."""
    names = sorted(n for n in os.listdir(indir) if n.endswith(".in"))
    cases = []
    for name in names:
        with open(os.path.join(indir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        cases.append((name[:-3], case))
    return cases


# ---------------------------------------------------------------------------
# History extraction
# ---------------------------------------------------------------------------

def extract_history(case):
    """Deduplicated reads: (t_hours, N, [rates or None]) oldest first.

    Consecutive reads that change nothing carry no information: the
    page is a stale snapshot between Exophase refreshes, so only the
    first read of each identical run constrains the fit.  (Keeping a
    later duplicate would force the N-curve flat through hours when
    the true N was growing unobserved.)
    """
    rows = []
    for read in case.get("history", []):
        try:
            t = hours_since_release(parse_time(read["at"]))
            n = float(read["players_tracked"])
        except (KeyError, TypeError, ValueError):
            continue
        rates = read.get("rates") or []
        rows.append((t, n, rates))
    rows.sort(key=lambda r: r[0])
    dedup = []
    for row in rows:
        if dedup:
            pt, pn, pr = dedup[-1]
            if row[1] == pn and row[2] == pr:
                continue
        dedup.append(row)
    return dedup


def last_rates(case, n_trophies):
    """Last non-null rate per trophy (fallback: 0.01)."""
    out = [EPS_RATE] * n_trophies
    for read in reversed(case.get("history", [])):
        rates = read.get("rates") or []
        if len(rates) < n_trophies:
            continue
        if all(r is not None for r in rates):
            return [float(r) for r in rates]
        for i, r in enumerate(rates):
            if r is not None and out[i] == EPS_RATE:
                out[i] = float(r)
        if all(v != EPS_RATE for v in out):
            break
    return out


# ---------------------------------------------------------------------------
# Players-tracked: log-logistic S-curve ensemble
# ---------------------------------------------------------------------------

def loglogistic_n(t, ninf, t50, beta):
    """N(t) = Ninf / (1 + (t50/t)**beta); t in hours from the time zero.

    A power law early (N ~ Ninf * (t/t50)**beta while t << t50) that
    saturates at Ninf -- it can fit the launch-night pattern, N up 6x
    in 4.5 h then 2.6x in the next 6.7 h, which a Weibull in clock
    time cannot, because its origin sits at release.
    """
    if t <= 0:
        return 0.0
    x = (t50 / t) ** beta
    if x > 700:
        return 0.0
    return ninf / (1.0 + x)


# Ceiling multiples over the last observed N; half-time and slope grids.
NINF_MULTS = (1.2, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 60.0, 100.0)
T50_GRID = (12.0, 20.0, 30.0, 45.0, 70.0, 110.0, 170.0, 260.0, 400.0,
            700.0, 1200.0)
BETA_GRID = (0.8, 1.5, 2.5, 4.0, 6.0, 9.0)
# Wide lognormal prior on the ceiling: median 25x the last N, one
# log-sigma ~ a factor of e.  Early data cannot separate a world that
# saturates at 250k from one that runs to 600k+, so the prior -- not
# the fit -- decides how much weight each world gets, and the mixture
# keeps both alive instead of betting the entry on one of them.
PRIOR_MED_MULT = 25.0
PRIOR_LOG_SIGMA = 1.0
FIT_LOG_SIGMA = 0.12
KEEP_SCENARIOS = 10


def fit_n_ensemble(times, ns):
    """Grid-fit the log-logistic N-curve; return [(weight, ninf, t50, beta)].

    Weights are a fit likelihood (SSE on log N) times the ceiling
    prior, normalised over the best KEEP_SCENARIOS grid points.
    """
    if not times:
        return [(1.0, 1000.0, 200.0, 1.0)]
    n_last = ns[-1]
    log_n = [math.log(max(n, 1.0)) for n in ns]
    scored = []
    for mult in NINF_MULTS:
        ninf = n_last * mult
        log_prior = -0.5 * ((math.log(mult / PRIOR_MED_MULT)
                             / PRIOR_LOG_SIGMA) ** 2)
        for t50 in T50_GRID:
            for beta in BETA_GRID:
                sse = 0.0
                for t, ln in zip(times, log_n):
                    pred = loglogistic_n(t, ninf, t50, beta)
                    sse += (math.log(max(pred, 1.0)) - ln) ** 2
                # Likelihood on log-N residuals; flat when reads are frozen.
                like = -sse / (2.0 * FIT_LOG_SIGMA ** 2 * max(len(times), 1))
                scored.append((like + log_prior, ninf, t50, beta))
    scored.sort(key=lambda s: -s[0])
    best = scored[0][0]
    kept = []
    total = 0.0
    for score, ninf, t50, beta in scored[:KEEP_SCENARIOS]:
        w = math.exp(max(score - best, -30.0))
        kept.append([w, ninf, t50, beta])
        total += w
    if total <= 0:
        return [(1.0, n_last * PRIOR_MED_MULT, 200.0, 1.0)]
    return [(w / total, ninf, t50, beta) for w, ninf, t50, beta in kept]


def n_at(scenario, t):
    _, ninf, t50, beta = scenario
    return loglogistic_n(t, ninf, t50, beta)


# ---------------------------------------------------------------------------
# Per-trophy earners ODE:  dE/dt = p * dN/dt + h * (N - E)
# ---------------------------------------------------------------------------

# Coarse grids for the first pass; the second pass refines x3 around
# the coarse winner, so the fit reaches fine-grid accuracy at a third
# of the cost -- what keeps a 2-day, 10-minute history inside 8 s.
P_COARSE = (0.0, 0.0005, 0.002, 0.01, 0.03, 0.1, 0.3, 1.0)
H_COARSE = (0.0, 0.0005, 0.002, 0.008, 0.032, 0.064)
# Prior centres: a brand-new tracked player rarely holds a hard trophy,
# but a trophy (nearly) everyone holds is probably held by joiners too,
# so the p-prior scales with the trophy's own last rate.  The hazard
# prior scales the same way: easy trophies keep being earned.
P_PRIOR = {"platinum": 0.0, "gold": 0.001, "silver": 0.005, "bronze": 0.01}
# Maturity lag (hours) for the hazard pool: only players tracked at
# least this long ago can have earned the trophy.  Off by default; the
# growth cap below does most of the regularising.
USE_LAG = False
POOL_LAG = {"platinum": 72.0, "gold": 48.0, "silver": 24.0, "bronze": 12.0}
# Half-life (hours) for weighting reads in the (p, h) fit: recent reads
# count more, since launch-week growth regimes change fast.  None = off.
DECAY_HALF_LIFE_H = None


def prior_ph(tier, last_rate):
    """(p0, h0) priors from tier and the trophy's last observed rate."""
    frac = clamp(last_rate, EPS_RATE, 100.0) / 100.0
    p0 = clamp(0.85 * frac ** 1.5 + 0.5 * P_PRIOR.get(tier, 0.01), 0.0, 0.95)
    h0 = 0.001 + 0.03 * frac
    return p0, h0


def n_obs_func(times, ns):
    """Observed N(t): log-linear interpolation between reads, clamped
    to the endpoints outside.  The (p, h) fit runs on this, never on a
    scenario curve, so a scenario that misses the observed N cannot
    corrupt the fit."""
    log_t = [math.log(max(t, 1e-6)) for t in times]
    log_n = [math.log(max(n, 1.0)) for n in ns]

    def f(t):
        lt = math.log(max(t, 1e-6))
        if lt <= log_t[0]:
            return ns[0]
        if lt >= log_t[-1]:
            return ns[-1]
        k = bisect.bisect_right(log_t, lt) - 1
        frac = (lt - log_t[k]) / (log_t[k + 1] - log_t[k] or 1.0)
        return math.exp(log_n[k] + frac * (log_n[k + 1] - log_n[k]))

    return f


def growth_cap(times, earners, n_last):
    """Fastest observed hourly earner gain, x3, with a floor.

    Bounds how fast earners may grow in any simulation, in-sample or
    forecast: a (p, h) that would flood the trophy with earners far
    faster than the history ever saw is cut down to that pace.
    """
    top = 0.0
    for k in range(len(times) - 1):
        dt = times[k + 1] - times[k]
        if dt > 0:
            top = max(top, (earners[k + 1] - earners[k]) / dt)
    return max(3.0 * top, 0.0001 * max(n_last, 1.0), 0.05)


def step_earners(t0, e0, t1, p, h, nfunc, cap=None, lag=0.0):
    """Integrate the ODE from t0 to t1 (hours) starting at earners e0.

    nfunc gives N(t); lag, when set, takes the hazard pool from
    N(t - lag) instead of N(t).  cap bounds earners gained per hour.
    """
    if t1 <= t0:
        return e0
    span = t1 - t0
    steps = max(1, min(48, int(span / 0.5) + 1))
    dt = span / steps
    t, e = t0, e0
    for _ in range(steps):
        n0 = nfunc(t)
        n1 = nfunc(t + dt)
        pool = nfunc(t - lag) - e if lag > 0 else n0 - e
        de = p * (n1 - n0) + h * max(pool, 0.0) * dt
        if cap is not None:
            de = min(de, cap * dt)
        e = min(max(e + de, 0.0), n1)
        t += dt
    return e


def simulate_rates(times, earners0, nfunc, p, h, cap=None, lag=0.0):
    """Predicted rate (%) at each time, starting from earners0 at times[0]."""
    out = []
    t_prev, e = times[0], earners0
    for t in times:
        e = step_earners(t_prev, e, t, p, h, nfunc, cap, lag)
        n = nfunc(t)
        out.append(100.0 * e / n if n > 0 else EPS_RATE)
        t_prev = t
    return out


def fit_trophy(times, earners, tier):
    """Best (p, h) on the logit scale, with shrinkage to rate priors.

    times/earners are the distinct reads; earners[0] anchors the ODE.
    The fit runs against the observed N path, so it is done once per
    trophy and never depends on which N scenario wins the ensemble.
    """
    if not times:
        return 0.01, 0.003, 0.0
    n_here = FIT_N[len(times) - 1] if len(FIT_N) >= len(times) else 0.0
    last_rate = 100.0 * earners[-1] / n_here if n_here > 0 else EPS_RATE
    p0, h0 = prior_ph(tier, last_rate)
    lag = POOL_LAG.get(tier, 0.0) if USE_LAG else 0.0
    if len(times) < 2:
        return p0, h0, 0.0
    obs = [logit(100.0 * e / n) if n > 0 else logit(EPS_RATE)
           for e, n in zip(earners, FIT_N)]
    nfunc = n_obs_func(times, FIT_N)
    cap = growth_cap(times, earners, FIT_N[-1])

    def search(p_grid, h_grid):
        best = None
        for p in p_grid:
            for h in h_grid:
                pred = simulate_rates(times, earners[0], nfunc, p, h,
                                      cap, lag)
                sse = 0.0
                for k, (a, b) in enumerate(zip(pred, obs)):
                    w = 1.0
                    if DECAY_HALF_LIFE_H:
                        w = 0.5 ** ((times[-1] - times[k])
                                    / DECAY_HALF_LIFE_H)
                    sse += w * (logit(a) - b) ** 2
                pen = 2.0 * ((p - p0) / 0.25) ** 2
                pen += 0.5 * (math.log((h + 1e-5) / h0)) ** 2
                score = sse + pen
                if best is None or score < best[0]:
                    best = (score, p, h, sse)
        return best

    # Coarse pass, then a x3 neighbourhood refinement around its winner:
    # the same accuracy as the full fine grid at a third of the cost,
    # which is what keeps a 2-day, 10-minute history inside 8 seconds.
    best = search(P_COARSE, H_COARSE)
    p_ref = sorted({clamp(best[1] / 3.0, 0.0, 1.0), best[1],
                    clamp(best[1] * 3.0, 0.0, 1.0)})
    h_ref = sorted({max(best[2] / 3.0, 0.0), best[2], best[2] * 3.0})
    refined = search(p_ref, h_ref)
    if refined[0] < best[0]:
        best = refined
    return best[1], best[2], best[3]


# Set by forecast_case before fitting: the N observed at each distinct
# read (parallel to times).
FIT_N = []


def calibrate_sigma(times, earners_by_idx, tiers, horizon_h):
    """Spread scale from rolling-origin one-step-ahead logit errors.

    Fit on reads up to each of the last few distinct origins, predict
    the next read, and turn the RMS logit error into a per-root-hour
    diffusion, then scale it to the horizon.  Falls back to a
    baseline-like width when the history has no second distinct read.
    """
    days = max(horizon_h, 1.0) / 24.0
    if len(times) < 3:
        return 0.5 * math.sqrt(days)
    gaps = [b - a for a, b in zip(times, times[1:])]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else 1.0
    gap = max(gap, 1.0 / 60.0)
    # Subsample trophies across the rate range so calibration stays fast.
    idxs = sorted(range(len(earners_by_idx)),
                  key=lambda i: earners_by_idx[i][-1])
    sample = idxs[::max(1, len(idxs) // 8)][:8] or idxs[:1]
    errs = []
    for origin in range(max(1, len(times) - 3), len(times) - 1):
        sub_t, sub_n = times[:origin + 1], FIT_N[:origin + 1]
        for i in sample:
            series = earners_by_idx[i]
            p, h, _ = fit_trophy(sub_t, series[:origin + 1], tiers[i])
            nfunc_o = n_obs_func(sub_t + [times[origin + 1]],
                                 FIT_N[:origin + 2])
            cap_o = growth_cap(sub_t, series[:origin + 1], FIT_N[origin])
            lag_o = POOL_LAG.get(tiers[i], 0.0) if USE_LAG else 0.0
            pred = simulate_rates(sub_t + [times[origin + 1]],
                                  series[0], nfunc_o, p, h, cap_o,
                                  lag_o)[-1]
            obs = series[origin + 1]
            n_obs = FIT_N[origin + 1]
            if n_obs > 0 and obs >= 0:
                errs.append(logit(pred) - logit(100.0 * obs / n_obs))
    if not errs:
        return 0.5 * math.sqrt(days)
    rms = math.sqrt(sum(e * e for e in errs) / len(errs))
    diffusion = rms / math.sqrt(gap)
    return clamp(diffusion * math.sqrt(max(horizon_h, 1.0)), 0.30, 2.5)


# ---------------------------------------------------------------------------
# Mixture quantiles + case forecast
# ---------------------------------------------------------------------------

# Rough normal masses at the five z-points, for the mixture points.
ZMASS = (0.05, 0.20, 0.50, 0.20, 0.05)


def weighted_quantile(points, level):
    """Smallest value whose cumulative weight reaches `level`."""
    pts = sorted(points)
    total = sum(w for _, w in pts)
    if total <= 0:
        return pts[0][0] if pts else EPS_RATE
    acc = 0.0
    for value, w in pts:
        acc += w
        if acc >= level * total:
            return value
    return pts[-1][0]


def forecast_case(case):
    """Return [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]."""
    global FIT_N
    trophies = case.get("trophies", [])
    n_tr = len(trophies)
    if n_tr == 0:
        return []
    history = extract_history(case)
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_h = max((target - at).total_seconds() / 3600.0, 0.0)
    t_target = hours_since_release(target)

    if not history or horizon_h <= 0:
        # No usable history, or a zero horizon: persistence + RW spread.
        last = last_rates(case, n_tr)
        spread = 0.5 * math.sqrt(max(horizon_h, 1.0) / 24.0)
        rows = []
        for i, tr in enumerate(trophies):
            base = logit(last[i])
            qs = [round(expit(base + z * spread), 4) for z in ZVALS]
            rows.append({"id": tr["id"], "q": qs})
        return rows

    times = [r[0] for r in history]
    ns = [r[1] for r in history]
    # Thin to <=48 reads for fitting speed (10-min reads over 2 days
    # would otherwise make the (p, h) grid crawl); keep the last read.
    if len(times) > 48:
        keep = sorted(set(list(range(0, len(times), len(times) // 48))
                          + [len(times) - 1]))
        times = [times[i] for i in keep]
        ns = [ns[i] for i in keep]
        history = [history[i] for i in keep]
    FIT_N = ns
    ensemble = fit_n_ensemble(times, ns)

    # Earners per trophy at each distinct read.
    earners_by_idx = []
    for i in range(n_tr):
        series = []
        for t, n, rates in history:
            r = rates[i] if i < len(rates) else None
            series.append((r if r is not None else EPS_RATE) * n / 100.0)
        earners_by_idx.append(series)

    tiers = [tr.get("tier", "bronze") for tr in trophies]
    sigma_base = calibrate_sigma(times, earners_by_idx, tiers, horizon_h)

    t_last = times[-1]
    n_last = ns[-1]
    rows = []
    for i, tr in enumerate(trophies):
        p, h, _ = fit_trophy(times, earners_by_idx[i], tiers[i])
        e_last = earners_by_idx[i][-1]
        cap = growth_cap(times, earners_by_idx[i], n_last)
        lag = POOL_LAG.get(tiers[i], 0.0) if USE_LAG else 0.0
        points = []
        for w_s, ninf, t50, beta in ensemble:
            scen = (w_s, ninf, t50, beta)
            nfunc = lambda t, s=scen: n_at(s, t)
            e_tgt = step_earners(t_last, e_last, t_target, p, h, nfunc,
                                 cap, lag)
            n_tgt = n_at(scen, t_target)
            med = 100.0 * e_tgt / n_tgt if n_tgt > 0 else EPS_RATE
            # Saturation thins the upper tail: once a scenario's N has
            # (nearly) reached its ceiling, no new players arrive to
            # lift rates, so the upside above the median shrinks while
            # the downside (a slower hazard) stays wide.
            grown = (n_tgt - n_last) / max(ninf - n_last, 1.0)
            grown = clamp(grown, 0.0, 1.0)
            sig_up = sigma_base * (1.0 - 0.6 * grown)
            sig_dn = sigma_base
            centre = logit(med)
            for z, mass in zip(ZVALS, ZMASS):
                sig = sig_dn if z < 0 else sig_up
                points.append((expit(centre + z * sig), w_s * mass))
        qs = [round(weighted_quantile(points, lv), 4) for lv in LEVELS]
        qs = sorted(max(min(q, MAX_RATE), EPS_RATE) for q in qs)
        rows.append({"id": tr["id"], "q": qs})
    return rows


def main(argv):
    indir, outdir = input_dir(), output_dir()
    os.makedirs(outdir, exist_ok=True)
    for case_id, case in load_cases(indir):
        rows = forecast_case(case)
        payload = {"trophies": rows}
        out_path = os.path.join(outdir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
