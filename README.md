# E-commerce Sales & Demand Forecasting

End-to-end data analysis and forecasting project built on the [Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) from UCI Machine Learning Repository.

The goal was to go from raw transactional data all the way to a working demand forecasting model — covering data cleaning, exploratory analysis, customer segmentation, and time series forecasting.

---

## Project Phases

| Phase | Topic | Status |
|---|---|---|
| 1 | Environment Setup & Project Scaffolding | Complete |
| 2 | Data Cleaning & Quality Checks | Complete |
| 3 | Exploratory Data Analysis | Complete |
| 4 | Cloud Setup & SQL Analytics (GCP + BigQuery) | Planned |
| 5 | Demand Forecasting (Prophet + XGBoost) | Complete |
| 6 | Power BI Dashboard | Planned |
| 7 | Storytelling & Portfolio Publishing | In Progress |

---

## What this project covers

- **Data cleaning** — handling nulls, cancelled orders, negative quantities, and price outliers across 1M+ rows
- **Exploratory data analysis** — revenue trends, seasonality, product and country breakdowns, RFM customer segmentation
- **Demand forecasting** — two models (Prophet and XGBoost) trained on daily revenue, evaluated on a 60-day held-out test set

---

## Dataset

- Source: UCI Machine Learning Repository — Online Retail II
- Transactions from a UK-based online gift and homeware wholesaler
- Period: December 2009 to December 2011
- ~1 million rows after loading both sheets

---

## Project structure

```
ecommerce-forecasting/
├── data/
│   ├── raw/                          # original dataset (not tracked in git)
│   └── processed/
│       ├── online_retail_cleaned.csv       # output of phase 2
│       ├── online_retail_eda_complete.csv  # output of phase 3
│       └── forecast_results.csv           # actual vs predicted (phase 5)
├── notebooks/
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   └── 05_demand_forecasting.ipynb
├── reports/
│   ├── project_report.md             # full findings document
│   └── figures/                      # all charts saved here
└── README.md
```

---

## Key findings

### Business context
The data comes from a B2B wholesale operation — most customers are small retailers buying in bulk. This context explains a lot of the patterns found in the data.

### EDA highlights
- Revenue peaks sharply in **November** — driven by B2B customers stocking up before the holiday retail season, not direct consumer demand
- **Saturday is the weakest sales day** by a large margin, consistent with a business that sells to other businesses
- Top product by revenue: **Regency Cakestand 3 Tier**
- After the UK, the largest markets are **EIRE, Netherlands, and Germany**
- RFM segmentation identified ~5,878 unique customers — the largest segment being **Potential Loyalists** (1,436 customers)

### Forecasting results

Both models were trained on data up to October 2011 and tested on the final 60 days.

| Model | MAE (GBP) | RMSE (GBP) | MAPE |
|---|---|---|---|
| Prophet | 14,286.34 | 25,827.77 | 23.36% |
| XGBoost | 13,214.90 | 25,151.83 | 27.87% |

XGBoost edged out Prophet on MAE and RMSE. Prophet had a lower MAPE, meaning it handled the percentage error better on lower-revenue days. Neither model dominates completely, which is a realistic outcome for noisy daily retail data.

The MAPE figures (23–28%) are higher than what you'd see on weekly or monthly aggregations — daily revenue has a lot of variance from individual large orders or cancellations. The trend and seasonality are captured well by both models.

### Feature importance (XGBoost)
The most important feature by far was **is_weekend** (0.574 importance score), followed by **dayofweek** (0.112) and **rolling_mean_7** (0.081).

This tells a clear story: the biggest driver of whether revenue will be high or low on a given day is simply whether it's a weekday or weekend — which is exactly what we'd expect from a B2B wholesaler. After that, short-term momentum (rolling 7-day average) matters more than longer-term lags.

---

## Tech stack

- **Python** — pandas, numpy, matplotlib, seaborn, plotly
- **Forecasting** — Prophet (Meta), XGBoost
- **Evaluation** — scikit-learn (MAE, RMSE, MAPE)
- **Notebooks** — Jupyter (classic)
- **Version control** — Git + GitHub

---

## How to run

1. Clone the repo
2. Install dependencies: `pip install pandas numpy matplotlib seaborn plotly prophet xgboost scikit-learn`
3. Download the dataset from UCI and place it in `data/raw/`
4. Run notebooks in order: `02` → `03` → `05`
