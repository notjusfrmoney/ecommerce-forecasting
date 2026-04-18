# Project Report — E-commerce Sales & Demand Forecasting

**Dataset:** Online Retail II, UCI Machine Learning Repository  
**Period covered:** December 2009 – December 2011  
**Tools:** Python, Pandas, Plotly, Prophet, XGBoost

---

## 1. Problem Statement

A UK-based online retailer sells giftware and homeware products to customers across multiple countries. The business has over a million transaction records but no structured view of revenue trends, customer behaviour, or future demand.

The goal of this project was to:
- Clean and structure the raw transactional data
- Identify patterns in revenue, products, and customer behaviour through EDA
- Segment customers using RFM analysis
- Build and evaluate demand forecasting models that can predict daily revenue

---

## 2. Dataset Overview

The raw data contained approximately 1,067,000 rows across two annual sheets (2009–2010 and 2010–2011). Each row represents one line item in an invoice — a single product in a single transaction.

Key columns: Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country

Before any analysis could happen, the data needed significant cleaning.

---

## 3. Data Cleaning (Phase 2)

**Issues found in the raw data:**

- **Missing Customer IDs** — around 20% of rows had no customer identifier. These were kept for revenue analysis but excluded from customer-level analysis like RFM.
- **Cancelled orders** — invoices starting with 'C' represent cancellations with negative quantities. These were removed as they don't represent actual sales.
- **Negative or zero prices** — some rows had a price of 0 or below, likely internal transfers or data errors. These were removed.
- **Price outliers** — some products had extreme unit prices (e.g. £13,000+). Rather than deleting these, an `IsPriceOutlier` flag was added so they could be analysed separately without distorting aggregations.

After cleaning, the dataset had roughly 824,000 rows — a reduction of about 23% from the raw data.

A `TotalPrice` column (Quantity × Price) was added as the primary revenue metric. Date features (Year, Month, DayOfWeek, Hour, WeekOfYear) were also extracted from InvoiceDate for use in EDA and modelling.

---

## 4. Exploratory Data Analysis (Phase 3)

### 4.1 Revenue Over Time

Monthly revenue showed a strong upward trend from December 2009 through to November 2011, with a sharp peak every November and a drop in December.

The November peak is consistent with B2B wholesale behaviour — retail shop owners place large bulk orders ahead of the holiday season so their shelves are stocked. By December, their orders have already come in.

### 4.2 Day of Week Pattern

Revenue by day of week showed a clear weekday dominance. Monday through Thursday are the highest-revenue days. Friday is slightly lower. Saturday and Sunday drop significantly — especially Saturday.

This pattern is a strong signal that the customer base is made up of businesses, not individual consumers. Consumer e-commerce typically sees weekend spikes. This business sees the opposite.

### 4.3 Top Products

The top product by total revenue was **Regency Cakestand 3 Tier**, followed by items like White Hanging Heart T-Light Holder and Jumbo Bag Red Retrospot. These are all giftware and home decoration items consistent with a wholesaler supplying gift shops.

### 4.4 Geographic Breakdown

The UK accounts for the large majority of revenue. Among international markets, the top three were:

1. EIRE (Ireland)
2. Netherlands
3. Germany

Average Order Value (AOV) varied significantly by country. Some smaller-volume markets had much higher AOV, suggesting they purchase in larger quantities per order.

### 4.5 RFM Customer Segmentation

RFM stands for Recency, Frequency, and Monetary value — a standard framework for segmenting customers based on their purchasing behaviour.

- **Recency** — how recently did they last buy?
- **Frequency** — how many times have they bought?
- **Monetary** — how much have they spent in total?

Each customer was scored on all three dimensions and placed into a segment. Results from ~5,878 customers:

| Segment | Count |
|---|---|
| Potential Loyalists | 1,436 |
| At Risk | 1,150 |
| Champions | 621 |
| Hibernating | 580 |
| Lost | 489 |

The largest segment being Potential Loyalists suggests there is a real opportunity for the business to convert engaged but not-yet-loyal customers into repeat buyers. The At Risk segment (1,150 customers) is also worth attention — these were once active buyers who have gone quiet.

---

## 5. Demand Forecasting (Phase 5)

### 5.1 Preparing the Time Series

The transaction-level data was aggregated into a daily revenue series. Days with no transactions (weekends, holidays) were filled in with a revenue of 0 to keep the series continuous — both Prophet and XGBoost require evenly-spaced input.

The final daily series spanned from December 2009 to December 2011, with 731 total days.

### 5.2 Train / Test Split

The last 60 days were held out as the test set. The model was trained on all data before that cutoff and evaluated only on the final 60 days — dates it had never seen during training.

A chronological split is essential for time series. A random split would allow the model to train on future dates and predict past dates, which makes results look artificially good and doesn't reflect real-world deployment.

- Training period: December 2009 – October 2011 (671 days)
- Test period: October 2011 – December 2011 (60 days)

### 5.3 Model 1 — Prophet

Prophet (developed by Meta) is designed specifically for business time series. It decomposes the series into:

- **Trend** — the overall direction of revenue over time
- **Weekly seasonality** — which days of the week tend to be higher or lower
- **Yearly seasonality** — which months or seasons repeat each year

The components plot showed Prophet correctly identified:
- A growing revenue trend through 2010 and into 2011
- A strong weekly pattern with weekdays higher than weekends
- A yearly cycle with November as the annual peak

These findings are consistent with what the EDA showed independently.

**Prophet test set performance:**
- MAE: £14,286.34
- RMSE: £25,827.77
- MAPE: 23.36%

### 5.4 Model 2 — XGBoost

XGBoost is a gradient boosting algorithm that works on tabular data. Unlike Prophet, it has no built-in understanding of time — so temporal patterns had to be encoded manually as features.

Three types of features were created:

**Calendar features** — dayofweek, month, quarter, year, dayofyear, weekofyear, is_weekend  
Encode what time of year or week each row belongs to. The model can learn that month=11 is always high, or that dayofweek=5 (Saturday) is always low.

**Lag features** — lag_1, lag_7, lag_14, lag_30  
What was revenue 1 day ago, 7 days ago (same weekday last week), 14 days ago, 30 days ago. Gives the model memory of recent performance.

**Rolling statistics** — rolling_mean_7, rolling_mean_14, rolling_std_7  
Average revenue over the past 7 and 14 days, and the standard deviation over 7 days. Captures momentum and volatility.

All lag and rolling features were shifted by 1 day to prevent data leakage — the current day's revenue was never used as a feature to predict itself.

**XGBoost test set performance:**
- MAE: £13,214.90
- RMSE: £25,151.83
- MAPE: 27.87%

### 5.5 Model Comparison

| Model | MAE (GBP) | RMSE (GBP) | MAPE |
|---|---|---|---|
| Prophet | 14,286.34 | 25,827.77 | 23.36% |
| XGBoost | 13,214.90 | 25,151.83 | 27.87% |

XGBoost had a lower MAE and RMSE — it was closer to the actual daily revenue on average and made fewer large errors. Prophet had a lower MAPE — it handled percentage error better, particularly on low-revenue days.

Neither model completely dominates the other. For a business use case where the cost of large individual errors is high (e.g. over-ordering inventory), XGBoost's lower RMSE would be preferred. If the business cares more about being directionally correct across all days including quiet ones, Prophet's lower MAPE is the better metric.

The MAPE figures of 23–28% are higher than what you'd typically see on weekly or monthly forecasts. Daily revenue in retail is inherently noisy — a single large B2B order can double or halve the revenue on any given day. Aggregating to a weekly level would reduce this noise significantly and likely produce MAPE in the 10–15% range.

### 5.6 Feature Importance (XGBoost)

| Feature | Importance |
|---|---|
| is_weekend | 0.574 |
| dayofweek | 0.112 |
| rolling_mean_7 | 0.081 |
| lag_7 | 0.045 |
| lag_1 | 0.029 |

**is_weekend dominated at 57.4% importance** — by far the most predictive single feature. This makes complete sense for a B2B wholesaler. The model learned that the single biggest predictor of whether revenue will be high or low on a given day is whether it is a weekday or weekend. This aligns directly with the EDA finding that Saturday was the weakest sales day.

After is_weekend and dayofweek, the next most important features were short-term momentum — the 7-day rolling average and the lag from the same weekday last week. This suggests that recent sales trends carry real predictive power for next-day demand.

---

## 6. Summary of Key Findings

| Finding | Detail |
|---|---|
| Business type | B2B wholesale — giftware and homeware |
| Peak revenue month | November (pre-holiday B2B stocking) |
| Weakest sales day | Saturday |
| Top product | Regency Cakestand 3 Tier |
| Top international markets | EIRE, Netherlands, Germany |
| Unique customers | ~5,878 |
| Largest customer segment | Potential Loyalists (1,436 customers) |
| Best forecasting model (MAE/RMSE) | XGBoost |
| Best forecasting model (MAPE) | Prophet |
| Strongest predictive feature | is_weekend (57.4% importance in XGBoost) |

---

## 7. Limitations and Next Steps

**Limitations:**
- Daily forecasting is noisy — weekly or monthly aggregation would produce better MAPE
- No external regressors were included (e.g. UK public holidays, marketing events)
- Prophet was not tuned with custom holiday calendars which could improve the November peak prediction
- XGBoost hyperparameters were not tuned via cross-validation

**Possible next steps:**
- Add UK public holidays as Prophet regressors
- Aggregate to weekly level and re-evaluate both models
- Try a per-product or per-country forecast rather than total revenue
- Add hyperparameter tuning with TimeSeriesSplit cross-validation
- Connect to a cloud dashboard (BigQuery + Power BI) for live monitoring
