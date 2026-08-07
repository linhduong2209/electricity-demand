# PROCESS — Working plan (Copilot scratchpad)

## Assessment

1. **Title vs implementation**: Current title "Impact of Weather Variables on Electricity Demand: A Comparative Machine Learning Study" under-sells the project. The project actually: (a) forecasts daily demand in Singapore using weather + calendar + autoregressive features, (b) compares 5 models, (c) uses PPSO hyperparameter optimisation, (d) ships a web portal (FastAPI BE + Vite FE). Feature importance shows lag/rolling features dominate → not purely "impact of weather". → Adjust title to forecasting-focused.
   - New title: "Weather-Aware Machine Learning for Short-Term Electricity Demand Forecasting in Singapore: A Comparative Study with PPSO Optimisation"
2. **PPSO fairness**: YES, unfair. Only CatBoost gets PPSO tuning; RF/XGB use fixed manual params. → Add generic `PPSOTunedModel` applying PPSO to RF, XGB, and CatBoost with TimeSeriesSplit CV fitness. Linear/baseline have no hyperparams → excluded (justified).
3. **Why PPSO** (from docs/1-s2.0-S2352710223006113): parameter-free (no inertia w, c1, c2 to hand-tune), phasor angle θ gives adaptive periodic control of exploration/exploitation, good for continuous non-convex expensive search spaces like hyperparam tuning; outperformed PSO/GA variants in the reference study.

## Steps

- [x] Read code (main.py, model.py, trainer.py, config.yaml) + report tex files
- [x] model.py: generic PPSO wrapper `PPSOTunedModel` for rf / xgb / catboost; keep `CatBoostPPSOModel` alias for backward compat (pickle for dashboard)
- [x] main.py: model_types = baseline, linear, rf, xgb, catboost, rf_ppso, xgb_ppso, catboost_ppso; per-model bounds; feature importance loop updated
- [x] config.yaml: add rf_ppso / xgb_ppso sections
- [x] Report edits (DTD_GroupReport):
  - [x] thesis.tex: new title + pdftitle
  - [x] design/design.tex: model list + PPSO applied to all tree models + why-PPSO justification
  - [x] impl/implementation.tex: PPSOTunedModel description
  - [x] eval/evaluation.tex: note new fair-comparison setup; metrics for rf_ppso/xgb_ppso marked TBD until re-run
  - [x] intro/abstract.tex + conclude.tex: wording "all tree-based models tuned with PPSO"
- [x] Run training (background) to regenerate leaderboard → update report numbers if finished (left running; user to update numbers)
- [x] RUN_GUIDE.md created — user will self-run training; awaiting new leaderboard numbers to fill TBD rows in eval/evaluation.tex (tab:model_comparison), then update abstract/conclusion numbers if best model changes
- [x] Delete: QUICK_START.md, status, test.py
- [x] Create README.md (project intro, train, BE, FE, structure)

## Notes

- Dashboard backend loads models/catboost_ppso_model.pkl → keep that filename.
- FINAL metrics (retrained, fair PPSO): catboost_ppso MAE 1496.2 / RMSE 1895.5 / R² 0.9135 / MAPE 0.89 / RAE 0.280 / WI 0.9774. Order by RMSE: catboost_ppso < catboost < linear < xgb_ppso < rf < rf_ppso < xgb. PPSO helps CB (−10.5% RMSE) & XGB (−6.4%), hurts RF (+2.1%).

## Compact paper restructure (done 3/8/2026)

- New structure (user request, 5–10 content pages): Introduction / Related Work / Methodology / Results / Discussion / Conclusion / References.
- New files: `DTD_GroupReport/paper/{introduction,relatedwork,methodology,results,discussion,conclusion}.tex` — condensed, theory-light (PPSO principles referenced not re-derived), emphasis on own contributions (fair tuning protocol, CCI feature, portal).
- `thesis.tex`: switched includes to `\input{paper/...}`; added geometry margin=2.2cm, \setstretch{1.15}, titlesec tighter heading spacing, section/figure/table numbering without chapter prefix. Old chapter files kept in repo but not included.
- Figures kept: pipeline tikz, PPSO workflow tikz, predictions plot, feature importance (catboost_ppso), 2 tables (comparison + ppso impact).
- All numbers updated to new leaderboard (also in old chapter files eval/abstract/conclude for consistency).
- ⚠ No local LaTeX (pdflatex not installed) → user must compile on Overleaf; check page count there.

## Round 2 polish (3/8/2026, later)

- Abstract page removed (`\include{intro/abstract}` dropped from thesis.tex; file kept in repo).
- Declaration page → "Declaration of Contributions": redefined `\declpage` in thesis.tex preamble with a 6-row contribution table (names left as \dotfill for user to fill; roles: data, features/EDA, models, PPSO, portal, evaluation/report).
- Added figures: `design/images/eda_time_series.png` (3-year demand+temperature series) in Methodology §Data; `impl/images/dashboard_history.png` in §Web Portal; new Table `tab:search_spaces` (PPSO search spaces per model).
- Shortened introduction.tex, relatedwork.tex, discussion.tex, methodology (CCI inline, no separate equation).
- README.md rewritten: full end-to-end guide incl. venv setup, pip install, tools/parse.py (raw EMA .xls → csv), tools/check.py (missing-date validation), train, BE, FE.
