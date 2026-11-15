# Code supplement — choosing the exponent of a fractional power basis in adaptive robust regression

Scripts, stored Monte Carlo replications and the Lean project behind the article
"Choosing the exponent of a fractional power basis in adaptive robust regression"
and its online supplement (Appendices A–F).

Archived on Zenodo: release v1.0.0 is
[doi:10.5281/zenodo.24241609](https://doi.org/10.5281/zenodo.24241609); the concept DOI
[10.5281/zenodo.24241608](https://doi.org/10.5281/zenodo.24241608) always resolves to the
latest version.

Every table of the article and of the supplement is produced from the result files stored
here; the two table scripts never rerun a simulation. Running them regenerates
`paper/tables/table1.tex` and `paper/supplement/tables/*.tex`, and the output matches the
files the manuscript was built from byte for byte.

## Layout

| Path | What it is | Where it is used |
|---|---|---|
| `paper/make_tables.py` | Table 1 and every number quoted in the article, from `results/letter_2026-11-13/e7_per_rep/` and `e5_per_rep/` | Table 1, Sections 4–5 |
| `paper/make_supplement.py` | the 13 tables of the supplement, from the stored result files | Appendices A–E |
| `scripts/table1_mc.py`, `scripts/baselines_asm.R` | E7, the consolidated run behind Table 1 (the R script runs the `asm` baseline on the saved data) | Table 1, Appendix B |
| `scripts/tau_patp_adaptive_mc.py`, `scripts/run_largeN.sh` | the data-driven exponent at large N for three generating elements (Huber, bisquare, Welsch) | Appendix C |
| `scripts/selector_bias_pop.py` | population calculation of the selector's lag and of the bias rate | Appendix C |
| `scripts/t9_check.py` | numerical check of Theorem 1 | Appendix A |
| `scripts/newey_identity_check.py` | the identity with Newey's (1988) estimator, for integer and fractional bases | Appendix D |
| `scripts/baselines_mc.py`, `scripts/newey_weighted_mc.py` | E4 and E4b, comparisons with prior adaptive estimators | Appendix D |
| `scripts/sensitivity_mc.py` | E5, sensitivity to the contamination model and to the bisquare constant | Appendix E |
| `scripts/tau_patp_mc.py`, `scripts/tau_patp_pop.py`, `scripts/generating_elements_mc.py`, `scripts/generating_elements_pop.py`, `scripts/poplaws.py` | exploratory runs on robust generating elements that preceded the article; their outputs are in `results/generating_2026-11-12/` | background only |
| `results/` | stored outputs: printed tables (`*.txt`) and per-replication arrays (`*_per_rep/`) | all |
| `results/letter_2026-11-13/selector/` | gate record of the Lean project (`gate.json`, `snapshot.json`) and the selector runs | Appendix A, C |
| `Lean/`, `lakefile.lean`, `lake-manifest.json`, `lean-toolchain` | Lean 4 project (v4.26.0, Mathlib v4.26.0): eight lemmas behind Theorem 2 and Proposition 3, each with a witness | Appendix A, F |

Comments in some scripts refer to working notes (`theory/*.tex`) that are not part of this
archive; their content is Appendix A of the online supplement.

## Reproduce

Python 3.13 with the versions in `requirements.txt`; R 4.5.3 with the package `asm` 0.2.4
for the `asm` baseline only.

```bash
python -m pip install -r requirements.txt
python paper/make_tables.py        # Table 1 and the quoted numbers, from stored results
python paper/make_supplement.py    # the 13 supplement tables, from stored results
```

`make_tables.py` also writes `paper/claims.yaml`, a ledger tying every number to its source;
outside the manuscript tree it reports that the printed forms were not found, which is expected.

To rerun a simulation, call the script with the arguments in its docstring, for example
`python scripts/table1_mc.py run results/new 2000 500,2000,8000` followed by
`Rscript scripts/baselines_asm.R results/new 8`. Seeds are fixed in each script
(`numpy.random.default_rng` with a seed sequence per law and sample size).

To check the Lean project:

```bash
lake exe cache get
lake build TauPATP
```

The build has no `sorry`; `#print axioms` reports only `propext`, `Classical.choice` and
`Quot.sound` for all sixteen theorems.

## License

MIT, see `LICENSE`.
