"""E5: sensitivity of the E7 arms to the contamination model and to the bisquare constant.

Usage:
  sensitivity_mc.py run <out_dir> <R> [N] [processes]
  sensitivity_mc.py collect <out_dir>

One factor is varied at a time around the E7 contaminated cell (5 % of errors replaced by
+-10 + N(0,1), bisquare constant 4.685), on three cores (Gaussian, Gamma(4), bimodal):
  frac   contamination fraction 0, 1, 2, 5, 10 % at +-10
  loc    outlier location 5, 7, 20 at 5 % (10 is the frac = 5 % cell)
  side   one-sided outliers: 5 % at +10, 10 % at +10, 5 % at +5
  const  bisquare constant x 0.7 and x 1.3 (pilot M-estimate and generating element alike),
         without contamination and at 5 % +-10
Arms and the estimator are those of table1_mc.one_rep (every arm on the same samples within a
cell). Seed: [20261121, cell index].
"""
import glob
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

import baselines_mc
from baselines_mc import GRID, P, bimodal, gamma4, gauss
from table1_mc import COLS, K0, one_rep, ratio_se

C_B0 = 4.685
CORES = [("Гаус", gauss), ("Gamma(4)", gamma4), ("бімодальний", bimodal)]


def cells():
    out = []
    for ci in range(len(CORES)):
        for frac in (0.0, 0.01, 0.02, 0.05, 0.10):
            out.append(("frac", ci, frac, 10.0, False, 1.0))
        for loc in (5.0, 7.0, 20.0):
            out.append(("loc", ci, 0.05, loc, False, 1.0))
        for frac, loc in ((0.05, 10.0), (0.10, 10.0), (0.05, 5.0)):
            out.append(("side", ci, frac, loc, True, 1.0))
        for mult in (0.7, 1.3):
            for frac in (0.0, 0.05):
                out.append(("const", ci, frac, 10.0, False, mult))
    return out


def draw(core, n, rng, frac, loc, one_sided):
    e = core(n, rng)
    if frac > 0:
        bad = rng.random(n) < frac
        k = int(bad.sum())
        centre = np.full(k, loc) if one_sided else rng.choice([-loc, loc], k)
        e[bad] = centre + rng.standard_normal(k)
    return e


def run_cell(args):
    idx, cell, out_dir, R, n = args
    axis, ci, frac, loc, one_sided, mult = cell
    baselines_mc.C_B = C_B0 * mult  # read by tau() and w_bi at call time
    rng = np.random.default_rng([20261121, idx])
    res = np.empty((R, len(COLS)))
    for k in range(R):
        x = rng.standard_normal((n, P - 1))
        e = draw(CORES[ci][1], n, rng, frac, loc, one_sided)
        res[k] = one_rep(np.column_stack([np.ones(n), x]), e)
    np.save(os.path.join(out_dir, f"e5_{idx}_{n}_{R}.npy"), res)
    return idx


def label(cell):
    axis, ci, frac, loc, one_sided, mult = cell
    if frac == 0:
        cont = "без викидів"
    else:
        cont = f"{100 * frac:.0f}% {'+' if one_sided else '±'}{loc:.0f}"
    return f"{CORES[ci][0]:12s} {cont:12s} c×{mult:.1f}"


def collect(out_dir):
    all_cells = cells()
    fixed = slice(K0, K0 + len(GRID))
    ia, im = COLS.index("adapt"), COLS.index("M")
    print("E5: slope MSE / OLS slope MSE; one Newton step from the bisquare M-estimate; "
          "arms as in E7, same samples within a cell")
    print("adapt/M = paired ratio of slope MSE with a delta-method standard error")
    head = (f"{'комірка':34s}{'R':>6s}{'M':>8s}{'a=1':>8s}{'best':>14s}{'adapt':>14s}"
            f"{'NwT6':>8s}{'NwW6':>8s}{'adapt/M':>16s}")
    last_axis = None
    order = ("frac", "loc", "side", "const")

    def key(p):
        i = int(os.path.basename(p).split("_")[1])
        return order.index(all_cells[i][0]), i

    for path in sorted(glob.glob(os.path.join(out_dir, "e5_*.npy")), key=key):
        idx, n, R = (int(v) for v in os.path.basename(path)[3:-4].split("_"))
        cell = all_cells[idx]
        if cell[0] != last_axis:
            last_axis = cell[0]
            print(f"\n=== {last_axis}, N = {n} ===")
            print(head)
        res = np.load(path)
        m = res.mean(0)
        rel = m / m[0]
        kb = int(np.argmin(m[fixed]))
        print(f"{label(cell):34s}{R:6d}{rel[im]:8.3f}{rel[K0 + len(GRID) - 1]:8.3f}"
              f"{rel[K0 + kb]:8.3f} [{GRID[kb]:5.3f}]{rel[ia]:8.3f} [{m[COLS.index('alpha')]:4.2f}]"
              f"{rel[COLS.index('NwT6')]:8.3f}{rel[COLS.index('NwW6')]:8.3f}"
              + "{:9.3f} ({:.3f})".format(*ratio_se(res[:, ia], res[:, im])))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        out_dir, R = sys.argv[2], int(sys.argv[3])
        n = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
        procs = int(sys.argv[5]) if len(sys.argv) > 5 else 4
        jobs = [(i, c, out_dir, R, n) for i, c in enumerate(cells())]
        with Pool(procs) as pool:
            for i in pool.imap_unordered(run_cell, jobs):
                print("cell", i, "done", flush=True)
    else:
        collect(sys.argv[2])
