# Pareto Design

The locked XGBoost models in `ML2/XGB.xlsx` are used for HV, EC and Q3. The workflow:

1. reproduces the fixed 80/20 split metrics;
2. generates a Sobol composition space with Cu by balance;
3. recalculates all physicochemical and CTD-derived features;
4. rejects candidates outside the Cu–Ni–Si–Cr applicability domain;
5. uses bootstrap lower-confidence predictions for the HV–EC Pareto front;
6. exports strength-anchor, balanced-knee and conductivity-anchor records.

Run `Pareto_Design_TRUNK.ipynb`. Before the formal run, replace the fixed aging condition with the real peak-aging condition and keep it identical for all candidates. The balanced knee (`P2_Balanced_knee`) is the recommended primary experimental alloy; P1 and P3 are boundary/reference alloys.

Do not claim that a single point is globally “the best” without stating the selection rule. In the manuscript use “model-selected balanced knee point on the robust Pareto front.”
