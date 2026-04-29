# DSA210 Term Project
# Momentum Collapse in Professional Tennis

## Project Overview
This project investigates whether professional tennis players experience a short-term performance decline after losing a break point. The main idea is that losing a critical point (break point) may negatively affect a player's performance in the following points, reflecting a temporary "momentum collapse".

---

## Data Sources

Two public datasets by Jeff Sackmann are used:
- **Point-by-point data** (`tennis_pointbypoint`): contains the sequence of points within each match
- **ATP match data** (`tennis_atp`): includes player rankings, surface type, tournament level, and match information

These datasets are merged to create an **event-level dataset** where each row represents a lost break point event.

---

## Methodology

### 1. Data Collection
- Extracted point-by-point match data from Jeff Sackmann's tennis datasets
- Identified break point situations (receiving player can win the game on the next point)
- Detected lost break point events (player was the receiver, break point score, player lost the point)

### 2. Feature Engineering
For each lost break point event:
- Calculated **PBPP (Post-Break-Point Performance)**:
  - Next 3 points (k=3)
  - Next 6 points (k=6)
  - Next 12 points (k=12)
- Computed **Performance Drop (PD)**:

  `PD = PBPP - baseline win rate`

### 3. Data Enrichment
Merged with ATP dataset to add:
- Surface (clay, hard, grass)
- Player ranking
- Opponent ranking
- Rank difference
- Tournament information

### 4. Data Cleaning
- Removed rows with missing PBPP values (occurs when fewer than k points remain in the match)
- Removed duplicate rows

---

## Exploratory Data Analysis (EDA)

- Distribution of PBPP (k=6)
- Distribution of Performance Drop (PD) — slightly shifted below zero, suggesting players perform worse than their baseline after losing a break point
- Average PBPP by surface
- Performance Drop (PD) boxplot by surface

---

## Hypothesis Testing

**Hypotheses tested:**
- H0: The mean performance drop after losing a break point is equal to zero
- H1: The mean performance drop after losing a break point is below zero

Tests were run for all three windows (k=3, k=6, k=12).

**Statistical tests used:**
- One-sample t-tests (for each k value)
- One-way ANOVA (to compare PD across surface types)

**Results:**
- All three t-tests produced statistically significant results (p < 0.05)
- Mean PD is below zero in all windows — players perform worse than their baseline after losing a break point
- The effect is **strongest at k=3** and weakens at k=6 and k=12, suggesting a short-term psychological impact
- Surface differences exist but are small

---

## Machine Learning

Building on the hypothesis testing results, supervised ML models were applied to predict whether a player will experience a performance drop (PD₃ < 0) after losing a break point.

**Target variable:** `collapse = 1` if PD₃ < 0, else 0  
(k=3 chosen because it showed the strongest and most statistically significant effect)

**Features used** (all available before the break-point event — no data leakage):

| Type | Features |
|---|---|
| Numerical | `player_rank`, `opponent_rank`, `rank_diff`, `baseline_win_rate`, `is_underdog`, `set_no`, `game_no` |
| Categorical (encoded) | `surface`, `pressure_level`, `round` |

**Models trained:**

| Model | Key Settings |
|---|---|
| Logistic Regression | `C=1.0`, `class_weight='balanced'`, StandardScaler applied |
| Decision Tree | `max_depth=5`, `min_samples_leaf=50`, `class_weight='balanced'` |
| Random Forest | `n_estimators=200`, `max_depth=10`, `class_weight='balanced_subsample'` |

**Evaluation method:** 80/20 stratified train/test split + 5-fold cross-validation  
**Metrics:** Accuracy, Precision, Recall, F1, ROC-AUC

**Key findings:**
- `baseline_win_rate` is the most predictive feature across all models — players who win fewer points overall are more likely to collapse
- `rank_diff` and `player_rank` also contribute significantly — higher-ranked players show more resilience
- `set_no` has some importance — collapse effects may intensify in later sets
- Random Forest achieved the best overall performance
- All models confirm that structural features carry predictive signal, but moderate AUC values indicate a genuine random/psychological component that pre-match features alone cannot fully capture
- ML results extend the hypothesis tests: moving from population-level inference to individual-level prediction of which players and contexts lead to collapse

---

## Results Summary

- A **significant performance drop** is observed immediately after losing a break point
- The effect is **strongest for k=3** and weakens for k=6 and k=12, confirming a short-term psychological impact
- Subgroup differences by surface and ranking group are small
- ML models confirm that player quality (baseline win rate, ranking) is the strongest structural predictor of whether a collapse occurs

---

## Repository Structure
analysis.ipynb               → main notebook (EDA, hypothesis testing, ML)
dsa210_tennis_pipeline.py    → full data preprocessing pipeline
breakpoint_events.csv        → event-level dataset (one row per lost break point)
matched_matches.csv          → enriched merged dataset
eda_summary.csv              → EDA summary statistics
hypothesis_tests.csv         → hypothesis test results
ml_model_comparison.csv      → ML model performance metrics
ProposalReport_Nil Uğur.pdf  → project proposal document
requirements.txt             → Python dependencies

---

## How to Run

1. Install dependencies:
pip install -r requirements.txt
2. Open and run the notebook:
analysis.ipynb

All cells can be run top to bottom. The notebook loads data directly from the GitHub raw URL.

---

## Notes

- Some PBPP values are missing when fewer than k points remain in the match — these rows are removed before hypothesis testing and ML
- The Decision Tree is trained to `max_depth=5` but only the first 3 levels are visualised for readability; gray nodes indicate further splits continue below
- ML target is binary (PD₃ < 0); a regression approach predicting the continuous PD value is a natural future extension
- Class imbalance is handled via `class_weight='balanced'` across all models

---

## Dependencies
pandas>=1.5
numpy>=1.23
scipy>=1.9
matplotlib>=3.6
scikit-learn>=1.2
seaborn>=0.12

---

## Author

Nil Uğur
