# Spotter Freight Rate ML Assessment - 2 to 3 Minute Loom Video Script
**Target Duration:** ~2:45 minutes  
**Format:** Camera ON, screen sharing the code and `candidate_december.png`.

---

### [0:00 - 0:35] Introduction, EDA Findings & Data Quality
> *"Hi everyone, I’m presenting my solution for the Spotter Freight Rate Prediction challenge.*
>
> *Starting with exploratory data analysis: our development data spans January through October 2025 across 48,000 loads, while the 12,000 validation loads cover November and December 2025.*
>
> *I identified three key data quality and domain considerations:*
> 1. *First, cargo weight had roughly 300 missing values. Rather than a naive global mean, I imputed weight by equipment type—using 31,000 lbs for Dry Van, 32,000 lbs for Reefer, and 30,000 lbs for Flatbed, along with missingness indicator flags.*
> 2. *Second, validation introduced 8 new cities not present in training. To prevent cold-start failures, I engineered coordinate-level spatial features—specifically Haversine distance, circuity ratios (road miles over great-circle miles), and directional deltas.*
> 3. *Third, exploring rate economics showed that `quote_signal` approximates an empirical rate-per-mile. Short-haul loads under 450 miles carry a ~32% premium per mile due to fixed terminal and loading time, whereas long-haul loads amortize those costs."*

---

### [0:35 - 1:15] Validation Strategy & Time Split Approach
> *"Turning to validation:*
>
> *Because the test set is strictly in the future, standard random K-Fold cross-validation would leak temporal macro patterns and yield dangerously optimistic metrics.*
>
> *To simulate reality, I implemented an Out-Of-Time (OOT) holdout split: training on January through August (38,477 loads) and validating on September and October (9,523 loads). This 2-month lookahead directly mirrors the gap between our training set and the November-December test horizon.*
>
> *We evaluated models on MAE, RMSE, and WAPE—Weighted Absolute Percentage Error—which is the industry standard for freight pricing because it avoids dividing by small dollar amounts."*

---

### [1:15 - 1:55] Model Selection & Benchmark Progression
> *"For modeling:*
>
> *We established a domain baseline using distance multiplied by quote signal and equipment ratios, which yielded an MAE of $246.29 and 10.29% WAPE.*
>
> *We then trained gradient-boosted decision trees: LightGBM optimizing an L1 (MAE) objective, and CatBoost with native categorical handling on lanes and equipment.*
>
> *Both models slashed error by over 54%: LightGBM reached $111.71 MAE (4.67% WAPE), and CatBoost reached $117.24 MAE.*
>
> *Our production champion is a 55/45 ensemble blend of LightGBM and CatBoost, which achieves optimal variance reduction, predicting with an MAE of $112.39 and 4.70% WAPE."*

---

### [1:55 - 2:35] Code Walkthrough & Fixed December Chart
> *(Screen switch to VS Code / code editor, highlighting `src/features.py` and `run_pipeline.py`)*
>
> *"Looking at the codebase:*
>
> *Everything is organized modularly under `src/`: `data.py` handles ingestion and imputation, `features.py` computes the spatial and holiday proximity signals, and `models.py` encapsulates our estimators.*
>
> *The entire pipeline is automated through `python run_pipeline.py`, which retrains the champion ensemble on all 48,000 loads, outputs the 12,000 predictions to `validation_predictions.csv`, and runs Spotter’s official `score.py` validator.*
>
> *(Screen switch to `scorer_results/candidate_december.png`)*
>
> *Here is the generated December chart for Lexington to Fort Wayne (360 miles, Dry Van). Notice two key economic patterns:*
> 1. *Weekly rhythm: clear midweek peaks on Wednesday/Thursday when freight shipping volumes peak, followed by weekend troughs when docks close.*
> 2. *Holiday capacity crunch: rates steadily build into Christmas week, peaking on December 25th at $844, followed by a month-end close at $845—all at a realistic $2.30/mile level."*

---

### [2:35 - 2:45] Conclusion & Verification
> *"Both `validation_predictions.csv` and the December chart passed all strict validations in `score.py` with zero errors. All code, requirements, and documentation are available in the repository. Thank you!"*
