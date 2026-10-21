# Observer-residual vs gradient-boosting detector for small abrupt faults

Model-based fault detection and isolation (FDI) for abrupt faults in process components is the setting studied in "Email:{eric. j. manders, gautam. biswas}@vanderbilt.edu" (2004). That work relies on an analytical model of the plant, and its residuals are only as good as that model. This experiment asks whether a data-driven classifier can do better when the nominal model is imperfect and the faults are small.

**Question:** Can a class-balanced gradient-boosting detector on windowed input-output features beat a model-based observer-residual detector for small abrupt parameter faults when the nominal model is imperfect?

**Dataset:** Simulated continuous-time mass-spring-damper episodes, integrated with `scipy.integrate.solve_ivp`. Each episode has random input excitation and measurement noise, plant parameters drawn around a nominal model with about 5% mismatch, and either no fault or an abrupt 5-30% step in stiffness or damping. Training uses seeds 0-9 and the test set uses separate seeds 100-109. This is not the paper's own dataset.

**Baseline:** A Luenberger observer built from the nominal parameters produces a residual. Windowed residual RMS is compared with a fixed threshold set at the 99th percentile of fault-free training windows.

**Improved model:** A class-balanced LightGBM classifier on sliding-window features (residual statistics, lagged input/output statistics). The decision threshold is chosen by grouped cross-validation on training episodes only. The test set is used once, at the end.

**How to run:**
```
pip install -r requirements.txt
python analysis.py
```

**Results:** Results are written to `results.json` (and `figures/f1_comparison.png`).

## Results (latest run)

On a simulated mass-spring-damper dataset (61,200 rows, held-out test episodes), a standard observer-residual threshold detector reached window-level F1 of 0.04, while a class-balanced LightGBM reached 0.45, though its false alarm rate rose to 36%.

| Metric | Value |
|---|---|
| baseline_score | 0.04159 |
| improved_score | 0.4537 |
| improvement | 0.4121 |
| baseline_score_std | 0 |
| improved_score_std | 0.007184 |
| baseline_far | 0.01083 |
| improved_far | 0.355 |
| baseline_small_fault_detection_rate | 0.01875 |
| improved_small_fault_detection_rate | 0.3931 |
| baseline_threshold | 0.02953 |
| improved_cv_threshold_mean | 0.4667 |
| improved_cv_f1_mean | 0.4297 |

Run on 2026-10-21. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Simulated data only; not the paper's dataset or plant. Train/test split is by episode seed (seeds 0-9 vs 100-109), not random. Windows overlap within episodes; a window is positive if the fault is active at its last sample. Baseline is deterministic, so its std is 0. Small sample of episodes; results depend on the simulated mismatch and noise levels. No paper result is reported for comparison.

## Reference

This project is motivated by the following paper:

> EJ Manders (2004). *Email:{eric. j. manders, gautam. biswas}@ vanderbilt. edu*. Fault Detection, Supervision and Safety of. https://books.google.com/books?hl=en&lr=&id=QAU9N34xOSsC&oi=fnd&pg=PA339&dq=gautam.biswas%40vanderbilt.edu&ots=ubXJYLS2zb&sig=szoxeJlk3l3i8bNkge_fuUKlPqw

Paper author: Prof. Gautam Biswas, Vanderbilt University (gautam.biswas@vanderbilt.edu)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
