# DSA210 Term Project

# Momentum Collapse in Professional Tennis

## Project Overview

This project investigates whether professional tennis players experience a short-term performance decline after losing a break point.

The main idea is that losing a critical point (break point) may negatively affect a player's performance in the following points, reflecting a temporary "momentum collapse".

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

## Results

- A **significant performance drop** is observed immediately after losing a break point
- The effect is:
  - **Strongest for k = 3**
  - **Weaker for k = 6 and k = 12**
- This suggests a **short-term psychological impact**
- Differences across surface and ranking groups are **small**

---

## Repository Structure

- `analysis.ipynb` → main analysis notebook  
- `breakpoint_events.csv` → event-level dataset  
- `matched_matches.csv` → enriched dataset  
- `eda_summary.csv` → summary statistics  
- `hypothesis_tests.csv` → statistical results  
- `requirements.txt` → dependencies  

---

## How to Run

1. Install dependencies: pip install -r requirements.txt
2. Open the notebook: analysis.ipynb
3. Run all cells

---

## Notes

- Some PBPP values are missing when fewer than k points remain in the match
- These rows are removed before hypothesis testing
- The full preprocessing pipeline was executed before generating the final dataset

---

## Author

Nil Uğur
