# DSA210 Term Project
# Momentum Collapse in Professional Tennis

## Project Overview
This project investigates whether professional tennis players experience a short-term performance decline after losing a break point. The main idea is that losing a critical point (break point) may negatively affect a player's performance in the following points, reflecting a temporary "momentum collapse".

---

## Dataset
Two public datasets are used:
- **Point-by-point data** (tennis_pointbypoint): contains the outcome of each point in a match
- **ATP match data** (tennis_atp): includes player rankings, surface type, and tournament information

These datasets are merged to create an **event-level dataset** where each row represents a lost break point event.

---

## Methodology

### 1. Data Collection
- Extracted point-by-point match data
- Identified break point situations
- Detected lost break point events

### 2. Feature Engineering
For each event:
- Calculated **PBPP (Post-Break-Point Performance)**:
  - Next 3 points
  - Next 6 points
  - Next 12 points
- Computed **Performance Drop (PD)**:

  PD = PBPP - baseline win rate

### 3. Data Enrichment
Merged with ATP dataset to add:
- Surface (clay, hard, grass)
- Player ranking
- Opponent ranking
- Tournament information

### 4. Data Cleaning
- Removed rows with missing PBPP values
- Removed duplicates

---

## Exploratory Data Analysis (EDA)
- Distribution of performance drop (PD)
- Comparison across:
  - Surface types
  - Ranking groups
- Summary statistics for all k values

---

## Hypothesis Testing
The following hypotheses were tested:
- H1: Players perform worse in the next 3 points after losing a break point
- H2: Players perform worse in the next 6 points
- H3: Players perform worse in the next 12 points

Statistical tests used:
- One-sample t-tests
- ANOVA (for surface and rank group comparisons)

---

## Machine Learning
Building on the hypothesis testing results, supervised ML models were applied to predict whether a player will experience a performance drop (PD₃ < 0) after losing a break point.

**Target variable:** `collapse = 1` if PD₃ < 0, else 0 (k=3 chosen as it showed the strongest effect)

**Features used:** `baseline_win_rate`, `player_rank`, `opponent_rank`, `rank_diff`, `is_underdog`, `surface`, `pressure_level`, `round`, `set_no`, `game_no`

**Models trained:**
| Model | Description |
|---|---|
| Logistic Regression | Linear probabilistic baseline with scaled features |
| Decision Tree | Interpretable rule-based classifier (max_depth=5) |
| Random Forest | Ensemble of 200 trees, best overall performance |

**Evaluation:** 80/20 stratified train/test split + 5-fold cross-validation. Models compared on Accuracy, Precision, Recall, F1, and ROC-AUC.

**Key findings:**
- `baseline_win_rate` is the most predictive feature across all models — players who win fewer points overall are more likely to collapse
- `rank_diff` and `player_rank` also contribute significantly — higher-ranked players show more resilience
- Random Forest achieved the best ROC-AUC; all models confirm that structural features carry predictive signal for momentum collapse

---

## Results Summary
- A **significant performance drop** is observed immediately after losing a break point
- The effect is **strongest for k = 3** and weakens for k = 6 and k = 12, suggesting a short-term psychological impact
- Differences across surface and ranking groups are small
- ML models confirm that player quality (ranking, baseline win rate) is the strongest predictor of whether a collapse occurs

---

## Repository Structure
- `analysis.ipynb` → main analysis notebook (EDA, hypothesis testing, and ML)
- `breakpoint_events.csv` → event-level dataset
- `matched_matches.csv` → enriched dataset
- `eda_summary.csv` → summary statistics
- `hypothesis_tests.csv` → statistical results
- `ml_model_comparison.csv` → ML model performance metrics
- `requirements.txt` → dependencies

---

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Open the notebook: `analysis.ipynb`
3. Run all cells

---

## Notes
- Some PBPP values are missing when fewer than k points remain in the match — these rows are removed before analysis
- The full preprocessing pipeline is in `dsa210_tennis_pipeline.py`
- ML target is binary (PD₃ < 0); regression on the continuous PD value is a possible future extension

---

## Author
Nil Uğur
