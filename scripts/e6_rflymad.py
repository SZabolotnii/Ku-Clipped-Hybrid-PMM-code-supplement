"""E6: the E7 arms with errors drawn from real barometer-minus-GPS residuals (RflyMAD).

Plan registered in Spec.md §9 (2026-11-15) before the run.

Errors: the baro - GPS residual of 50 real no-fault RflyMAD flights (Le et al. 2025,
doi:10.1177/02783649241305153; non-commercial licence), processed as in Ku-Baro-PATP exp3:
minus a 101-sample (20 s) moving median, divided by the flight's robust scale 1.4826 * MAD.
All flights are pooled; each replicate draws N errors from the pool independently, with
replacement. That removes the autocorrelation, so the Letter's model (iid errors independent of
x) holds exactly; only the shape of the noise is real.

Regression and arms as in E7 (table1_mc.one_rep, unchanged): x ~ N(0, I), columns OLS, M,
every fixed alpha, adaptive alpha, NwT6, NwW6.

Usage:
  e6_rflymad.py pool <resid_cache.npz> <out_dir>      build the pooled residuals (e6_pool.npy)
  e6_rflymad.py run <out_dir> <R> <N list> [max N whose data are saved for asm]
  e6_rflymad.py collect <out_dir>
"""
import glob
import math
import os
import sys

import numpy as np

from baselines_mc import GRID, P
from table1_mc import COLS, K0, one_rep, ratio_se

LI = 9  # law index used in file names (after the nine E7 laws), so asm_<li>_... matches E7


def moving_median(x, w):
    h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def build_pool(cache, out_dir):
    flights = np.load(cache, allow_pickle=True)["flights"]
    parts = []
    for f in flights:
        r = np.asarray(f["r"], dtype=float)
        rd = r - moving_median(r, 101)
        s = 1.4826 * np.median(np.abs(rd - np.median(rd)))
        parts.append(rd / s)
    pool = np.concatenate(parts)
    np.save(os.path.join(out_dir, "e6_pool.npy"), pool)
    m, sd = pool.mean(), pool.std()
    print(f"flights {len(parts)}, pooled residuals {len(pool)}")
    print(f"mean {m:.4f}  sd {sd:.4f}  skewness {((pool - m) ** 3).mean() / sd ** 3:.3f}  "
          f"excess kurtosis {((pool - m) ** 4).mean() / sd ** 4 - 3:.3f}")
    q = np.quantile(pool, [0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])
    print("quantiles 0.1%, 1%, 25%, 50%, 75%, 99%, 99.9%:", " ".join(f"{v:.3f}" for v in q))
    print(f"share |e| > 5: {np.mean(np.abs(pool) > 5):.4f}")


def run(out_dir, R, ns, save_max_n):
    pool = np.load(os.path.join(out_dir, "e6_pool.npy"))
    for n in ns:
        rng = np.random.default_rng([20261122, n])
        data = np.empty((R, n, P), dtype=np.float32) if n <= save_max_n else None
        res = np.empty((R, len(COLS)))
        for k in range(R):
            x = rng.standard_normal((n, P - 1)).astype(np.float32).astype(float)
            e = pool[rng.integers(0, len(pool), n)].astype(np.float32).astype(float)
            if data is not None:
                data[k, :, : P - 1] = x
                data[k, :, P - 1] = e
            res[k] = one_rep(np.column_stack([np.ones(n), x]), e)
        if data is not None:
            data.tofile(os.path.join(out_dir, f"data_{LI}_{n}_{R}.bin"))
        np.save(os.path.join(out_dir, f"t1_{LI}_{n}_{R}.npy"), res)
        print("E6", n, "done", flush=True)


def collect(out_dir):
    fixed = slice(K0, K0 + len(GRID))
    print("E6: slope MSE / OLS slope MSE; errors resampled iid from real RflyMAD baro - GPS residuals")
    print("criterion (registered): adapt / M paired ratio + 2 SE < 1 at some N >= 2000\n")
    print(f"{'N':>6s}{'R':>6s}{'M':>8s}{'best':>14s}{'a=1':>8s}{'adapt':>14s}{'NwT6':>8s}{'NwW6':>8s}{'asm':>8s}")
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, f"t1_{LI}_*.npy")),
                       key=lambda p: int(os.path.basename(p).split("_")[2])):
        _, _, n, R = os.path.basename(path)[:-4].split("_")
        n, R = int(n), int(R)
        res = np.load(path)
        apath = os.path.join(out_dir, f"asm_{LI}_{n}_{R}.txt")
        asm = np.genfromtxt(apath, skip_header=1)[:, 0] if os.path.exists(apath) else None
        m = res.mean(0)
        rel = m / m[0]
        kb = int(np.argmin(m[fixed]))
        line = (f"{n:6d}{R:6d}{rel[1]:8.3f}{rel[K0 + kb]:8.3f} [{GRID[kb]:5.3f}]"
                f"{rel[K0 + len(GRID) - 1]:8.3f}{rel[COLS.index('adapt')]:8.3f} "
                f"[{m[COLS.index('alpha')]:4.2f}]{rel[COLS.index('NwT6')]:8.3f}{rel[COLS.index('NwW6')]:8.3f}")
        if asm is not None:
            ok = ~np.isnan(asm)
            line += f"{asm[ok].mean() / res[ok, 0].mean():8.3f}"
        else:
            line += f"{'-':>8s}"
        print(line)
        rows.append((n, res, asm, kb))
    print("\nAll fixed alpha (slope MSE / OLS):")
    print(f"{'N':>6s}" + "".join(f"{a:>8.3f}" for a in GRID))
    for n, res, asm, kb in rows:
        m = res.mean(0)
        print(f"{n:6d}" + "".join(f"{v:8.3f}" for v in m[fixed] / m[0]))
    print("\nPaired ratios, adaptive / other arm (delta-method SE):")
    print(f"{'N':>6s}{'/ M':>16s}{'/ best fixed':>16s}{'/ a=1':>16s}{'/ NwT6':>16s}{'/ NwW6':>16s}{'/ asm':>16s}")
    verdict = False
    for n, res, asm, kb in rows:
        ad = res[:, COLS.index("adapt")]
        others = [res[:, 1], res[:, K0 + kb], res[:, K0 + len(GRID) - 1], res[:, COLS.index("NwT6")],
                  res[:, COLS.index("NwW6")]]
        line = f"{n:6d}" + "".join("{:9.3f} ({:.3f})".format(*ratio_se(ad, o)) for o in others)
        if asm is not None:
            ok = ~np.isnan(asm)
            line += "{:9.3f} ({:.3f})".format(*ratio_se(ad[ok], asm[ok]))
        print(line)
        r, se = ratio_se(ad, res[:, 1])
        if n >= 2000 and r + 2 * se < 1:
            verdict = True
    print(f"\nregistered criterion met: {verdict}")


if __name__ == "__main__":
    if sys.argv[1] == "pool":
        build_pool(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), [int(v) for v in sys.argv[4].split(",")],
            int(sys.argv[5]) if len(sys.argv) > 5 else 2000)
    elif sys.argv[1] == "collect":
        collect(sys.argv[2])
