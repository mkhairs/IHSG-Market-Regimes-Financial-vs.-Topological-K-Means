# IHSG Market Regime Analysis Using TDA and K-Means

## Overview

This notebook explores market regimes in the Jakarta Composite Index (IHSG) by combining financial time-series features, Topological Data Analysis (TDA), and K-Means clustering. The goal is to identify periods with different market conditions, such as normal activity, stress, and crisis-like behavior.

## Workflow

1. **Data collection and preparation** — obtain IHSG and relevant market data, align dates, and prepare the time series for analysis.
2. **Feature engineering** — derive market indicators such as returns, volatility, momentum, and drawdown to represent price behavior.
3. **Rolling-window analysis** — compute features over sequential windows to capture changes in market conditions through time.
4. **Topological feature extraction** — use TDA to summarize the shape and structure of local market dynamics.
5. **Feature scaling and clustering** — scale the features and apply K-Means to group observations with similar characteristics.
6. **Cluster evaluation** — assess candidate clustering configurations with internal metrics and stability-oriented comparisons.
7. **Regime interpretation and visualization** — map clusters back to the timeline and inspect whether they correspond to baseline, stress, or critical market periods.

## Expected Outputs

- A time-indexed dataset of engineered financial and topological features.
- Cluster assignments representing distinct market regimes.
- Evaluation metrics for comparing clustering configurations.
- Visualizations that show regime changes alongside IHSG movements.

## Interpretation Notes

- Cluster IDs are arbitrary; their meanings must be assigned after inspecting the corresponding feature profiles and time periods.
- A cluster indicates similarity in observed features, not a guaranteed future crash.
- Results depend on window size, feature selection, scaling, and the number of clusters.
- If the objective is early warning, the evaluation should distinguish predicting future stress from identifying stress that is already occurring.

## Scope

This Markdown file is a concise, code-free overview of the notebook's purpose and analysis workflow. It does not reproduce implementation details or cell outputs.
