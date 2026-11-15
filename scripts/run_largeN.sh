#!/bin/sh
# Large-sample adaptive-alpha runs for three generating elements, in parallel.
# Output: results/generating_2026-11-12/mc_adaptive_largeN_<tau>.txt
cd "$(dirname "$0")" || exit 1
PY=${PYTHON:-python3}
OUT=../results/generating_2026-11-12
"$PY" tau_patp_adaptive_mc.py 1000 1.0 "Гаус (,Лаплас,GG,t3" Huber 2000,8000,32000 1000,600,300 > "$OUT/mc_adaptive_largeN_Huber.txt" 2>&1 &
"$PY" tau_patp_adaptive_mc.py 1000 1.0 "Гаус (,Лаплас,GG,t3" bisquare 2000,8000,32000 1000,600,300 > "$OUT/mc_adaptive_largeN_bisquare.txt" 2>&1 &
"$PY" tau_patp_adaptive_mc.py 1000 1.0 "Гаус (,Лаплас,GG,t3" Welsch 2000,8000,32000 1000,600,300 > "$OUT/mc_adaptive_largeN_Welsch.txt" 2>&1 &
wait
echo done
