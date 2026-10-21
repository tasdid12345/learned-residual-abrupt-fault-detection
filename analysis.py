"""Baseline vs improved detector for small abrupt faults in a simulated
mass-spring-damper. Baseline: Luenberger observer residual with a fixed
99th-percentile threshold. Improved: class-balanced LightGBM on sliding-window
residual and input/output features, with the decision threshold chosen by
grouped cross-validation on training episodes only."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete, place_poles
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

SEEDS = [42, 43, 44]                    # model seeds (LightGBM is randomized)
DT, T_END, T_FAULT = 0.05, 40.0, 20.0   # sampling step, episode length, fault onset (s)
N = int(round(T_END / DT))              # samples per episode
ONSET = int(round(T_FAULT / DT))        # index of first faulty sample
W, STRIDE = 40, 5                       # window: 2 s long, 0.25 s stride
M0, C0, K0 = 1.0, 0.5, 2.0              # nominal model used by the observer
MISMATCH, NOISE, EPS_PER_SEED = 0.05, 0.02, 20


def simulate(rng):
    """One episode: plant params drawn around nominal (~5% mismatch), random
    piecewise-constant input, optional abrupt stiffness/damping fault of 5-30%."""
    m, c, k = (p * (1 + MISMATCH * rng.standard_normal()) for p in (M0, C0, K0))
    faulty = rng.random() < 0.5
    mag, sign, par = 0.0, 1, 0
    if faulty:
        mag = rng.uniform(0.05, 0.30)
        sign = rng.choice([-1, 1])
        par = rng.integers(2)           # 0 = stiffness, 1 = damping
    c2 = c * (1 + sign * mag) if (faulty and par == 1) else c
    k2 = k * (1 + sign * mag) if (faulty and par == 0) else k
    u_hold = rng.uniform(-1, 1, int(T_END))  # new input value every 1 s

    def rhs(t, x, cc, kk):
        u = u_hold[min(int(t), len(u_hold) - 1)]
        return [x[1], (u - cc * x[1] - kk * x[0]) / m]

    t = np.arange(N) * DT
    opts = dict(max_step=0.05, rtol=1e-6, atol=1e-8, dense_output=True)
    s1 = solve_ivp(rhs, (0, T_FAULT), [0.0, 0.0], args=(c, k), **opts)
    s2 = solve_ivp(rhs, (T_FAULT, T_END), s1.y[:, -1], args=(c2, k2), **opts)
    x = np.hstack([s1.sol(t[:ONSET]), s2.sol(t[ONSET:])])
    y = x[0] + NOISE * rng.standard_normal(N)          # noisy position sensor
    u = u_hold[np.minimum(t.astype(int), len(u_hold) - 1)]
    lab = np.zeros(N, dtype=int)
    lab[ONSET:] = int(faulty)
    return dict(y=y, u=u, lab=lab, faulty=bool(faulty), mag=mag)


def make_split(seed_offset):
    """Episodes from seeds seed_offset..seed_offset+9 (separate plant draws per seed)."""
    eps = []
    for s in range(10):
        rng = np.random.default_rng(seed_offset + s)
        eps += [simulate(rng) for _ in range(EPS_PER_SEED)]
    return eps


# Luenberger observer built from the NOMINAL parameters (imperfect model).
Ac = np.array([[0.0, 1.0], [-K0 / M0, -C0 / M0]])
Bc = np.array([[0.0], [1.0 / M0]])
Cc = np.array([[1.0, 0.0]])
Ad, Bd = cont2discrete((Ac, Bc, Cc, np.zeros((1, 1))), DT)[:2]
L = place_poles(Ad.T, Cc.T, [0.80, 0.70]).gain_matrix.T   # observer gain (2x1)


def residual(ep):
    """Observer residual r_k = y_k - C xhat_k."""
    xh, r = np.zeros(2), np.zeros(N)
    for k in range(N):
        r[k] = ep["y"][k] - xh[0]
        xh = Ad @ xh + Bd[:, 0] * ep["u"][k] + L[:, 0] * r[k]
    return r


def win(a, idx):
    """Windows of length W starting at idx."""
    return np.lib.stride_tricks.sliding_window_view(a, W)[idx]


def build(eps):
    """Window features. Column 2 is the residual RMS used by the baseline.
    Label = 1 if the fault is active at the window's last sample."""
    out = {k: [] for k in ["X", "y", "faulty", "mag", "grp"]}
    idx = np.arange(0, N - W + 1, STRIDE)
    h = W // 2
    for i, ep in enumerate(eps):
        R, Yw, U = win(residual(ep), idx), win(ep["y"], idx), win(ep["u"], idx)
        Rc = R - R.mean(1, keepdims=True)
        X = np.column_stack([
            R.mean(1), R.var(1), np.sqrt((R ** 2).mean(1)), np.abs(R).max(1),   # residual
            Yw.mean(1), Yw.var(1), U.mean(1), U.var(1),                         # I/O stats
            R[:, h:].mean(1) - R[:, :h].mean(1),                                # lagged residual trend
            Yw[:, h:].mean(1) - Yw[:, :h].mean(1),                              # lagged output trend
            (Rc[:, :-1] * Rc[:, 1:]).mean(1),                                   # residual lag-1 autocov
        ])
        n = len(idx)
        out["X"].append(X)
        out["y"].append(ep["lab"][idx + W - 1])
        out["faulty"].append(np.full(n, ep["faulty"]))
        out["mag"].append(np.full(n, ep["mag"]))
        out["grp"].append(np.full(n, i))
    return {k: np.concatenate(v) for k, v in out.items()}


PARAMS = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, class_weight="balanced", verbose=-1)


def fit_improved(tr, seed):
    """Class-balanced LightGBM. Threshold = F1-optimal on grouped 5-fold
    out-of-fold probabilities of the TRAINING episodes only."""
    oof = np.zeros(len(tr["y"]))
    for a, b in GroupKFold(5).split(tr["X"], tr["y"], tr["grp"]):
        m = lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(tr["X"][a], tr["y"][a])
        oof[b] = m.predict_proba(tr["X"][b])[:, 1]
    grid = np.linspace(0.05, 0.95, 19)
    f1s = np.array([f1_score(tr["y"], oof >= t) for t in grid])
    thr = grid[f1s.argmax()]
    model = lgb.LGBMClassifier(random_state=seed, **PARAMS).fit(tr["X"], tr["y"])
    return model, thr, f1s.max()


def evaluate(pred, d):
    """Main: window F1. Secondary: false-alarm rate on true-negative windows,
    detection rate on positive windows of faults under 10% magnitude."""
    pos = d["y"] == 1
    small = pos & (d["mag"] > 0) & (d["mag"] < 0.10)
    return dict(f1=f1_score(d["y"], pred), far=pred[~pos].mean(),
                dr_small=pred[small].mean() if small.any() else np.nan)


def main():
    t0 = time.time()
    print("Simulating episodes (train seeds 0-9, test seeds 100-109)...")
    tr, te = build(make_split(0)), build(make_split(100))
    print(f"train windows={len(tr['y'])}  test windows={len(te['y'])}")

    # Baseline: fixed threshold = 99th percentile of residual RMS on fault-free training windows.
    thr_b = np.percentile(tr["X"][~tr["faulty"], 2], 99)
    base = evaluate((te["X"][:, 2] > thr_b).astype(int), te)  # deterministic
    print(f"baseline: threshold={thr_b:.4f} test F1={base['f1']:.3f}")

    runs = []
    for seed in SEEDS:
        model, thr, cv_f1 = fit_improved(tr, seed)
        pred = (model.predict_proba(te["X"])[:, 1] >= thr).astype(int)
        r = evaluate(pred, te)
        r.update(cv_thr=float(thr), cv_f1=float(cv_f1))
        runs.append(r)
        print(f"improved seed={seed}: CV thr={thr:.2f} CV F1={cv_f1:.3f} test F1={r['f1']:.3f}")

    imp = {k: float(np.nanmean([r[k] for r in runs])) for k in ["f1", "far", "dr_small"]}
    imp_std = float(np.std([r["f1"] for r in runs]))
    base_std = float(np.std([base["f1"]] * len(SEEDS)))  # baseline has no randomness
    if not (np.isfinite(imp["f1"]) and te["y"].sum() > 0 and (te["y"] == 0).sum() > 0):
        print("Could not produce meaningful metrics.")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.bar(["baseline", "improved"], [base["f1"], imp["f1"]],
            yerr=[base_std, imp_std], capsize=5, color=["gray", "steelblue"])
    plt.ylabel("window-level F1 (test)")
    plt.title("Small-fault detection")
    plt.tight_layout()
    plt.savefig("figures/f1_comparison.png", dpi=120)
    plt.close()

    results = {
        "question": "Can a class-balanced gradient-boosting detector on windowed input-output "
                    "features beat a model-based observer-residual detector for small abrupt "
                    "parameter faults when the nominal model is imperfect?",
        "method": "Simulated mass-spring-damper (solve_ivp), random input and noise, abrupt "
                  "5-30% stiffness/damping faults, ~5% plant mismatch; baseline Luenberger "
                  "residual threshold vs class-balanced LightGBM on sliding-window features.",
        "dataset": {
            "name": "Simulated mass-spring-damper episodes (scipy solve_ivp)",
            "source_url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html",
            "n_rows": int(len(tr["y"]) + len(te["y"])),
            "n_train": int(len(tr["y"])),
            "n_test": int(len(te["y"])),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "window-level F1 for fault detection",
        "higher_is_better": True,
        "baseline": "Luenberger observer built from nominal parameters; residual RMS over a "
                    "2 s window compared with a fixed threshold (99th percentile of fault-free "
                    "training windows).",
        "improved": "Class-balanced LightGBM classifier on sliding-window residual and "
                    "input/output features; decision threshold chosen by grouped 5-fold CV "
                    "on training episodes.",
        "metrics": {
            "baseline_score": float(base["f1"]),
            "improved_score": imp["f1"],
            "improvement": imp["f1"] - float(base["f1"]),
            "baseline_score_std": base_std,
            "improved_score_std": imp_std,
            "baseline_far": float(base["far"]),
            "improved_far": imp["far"],
            "baseline_small_fault_detection_rate": float(base["dr_small"]),
            "improved_small_fault_detection_rate": imp["dr_small"],
            "baseline_threshold": float(thr_b),
            "improved_cv_threshold_mean": float(np.mean([r["cv_thr"] for r in runs])),
            "improved_cv_f1_mean": float(np.mean([r["cv_f1"] for r in runs])),
        },
        "notes": "Simulated data only; not the paper's dataset or plant. Train/test split is by "
                 "episode seed (seeds 0-9 vs 100-109), not random. Windows overlap within "
                 "episodes; a window is positive if the fault is active at its last sample. "
                 "Baseline is deterministic, so its std is 0. Small sample of episodes; "
                 "results depend on the simulated mismatch and noise levels. No paper result "
                 "is reported for comparison.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Done in {time.time() - t0:.1f}s. Wrote results.json")


if __name__ == "__main__":
    main()
