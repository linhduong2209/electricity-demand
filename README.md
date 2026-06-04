# Singapore Energy Demand Forecast

## Machine Learning Project for Next-Day Electricity Demand Prediction

### Project Overview

This project predicts Singapore's next-day electricity demand in MWh using historical weather conditions as the primary input features. Because air-conditioning is the dominant driver of electricity consumption in Singapore's tropical climate, weather variables such as temperature, humidity, and rainfall have a strong and learnable relationship with daily demand.

**Location:** Singapore (1.3521°N, 103.8198°E)  
**Data Sources:** Open-Meteo (free weather API) + Singapore Energy Market Authority (EMA)  
**Training Period:** 2023–2024 (~730 daily records)  

---

## Project Structure

```
electricity-demand/
├── singapore_energy_forecast.ipynb    # Main Jupyter Notebook (Complete Pipeline)
├── README.md                           # This file
```

---

## 9-Step Pipeline

### **Step 1: Data Collection**

- Fetch Open-Meteo daily weather data (temp, humidity, precipitation, wind speed)
- Load Singapore EMA half-hourly demand data, aggregate to daily MWh
- Merge datasets on date column

### **Step 2: Exploratory Data Analysis (EDA)**

- Time series visualization of demand and weather patterns
- Correlation analysis (temperature-demand relationship: r ≈ 0.82)
- Scatter plots and boxplots by day type
- Identify seasonal and weekly patterns

### **Step 3: Data Cleaning & Temporal Features**

- Add Singapore public holiday flags (2023-2024)
- Create temporal features: month, day_of_year, is_weekend, is_holiday
- Handle missing values
- Ensure 2023-2024 date coverage

### **Step 4: Feature Engineering**

- **Lag Features:** demand_lag_1, demand_lag_7, temp_lag_1
- **Rolling Statistics:** 7-day, 30-day, 3-day rolling averages
- **Interaction Features:** temperature × humidity
- **Total Features:** 20 engineered features

### **Step 5: Train/Test Split**

- Chronological split: first 80% for training, last 20% for testing
- Train: Jan 31 - Aug 20, 2024 (583 days)
- Test: Aug 21 - Dec 31, 2024 (147 days)
- NO shuffling to preserve time-series integrity

### **Step 6: Model Training**

- **Regression Models:**
  - Linear Regression (baseline)
  - Random Forest Regressor (best model)
- **Classification Models:**
  - Logistic Regression (baseline)
  - Random Forest Classifier (best model)
- Classification task: Predict demand level (High/Medium/Low) based on training set percentiles

### **Step 7: Model Evaluation**

- Regression metrics: MAE, RMSE, R²
- Classification metrics: Accuracy, F1 Score, Confusion Matrix
- Compare models against naive lag-1 baseline
- Visualize predictions vs actual demand

### **Step 8: Feature Analysis**

- Feature importance from Random Forest models
- Correlation matrix
- Impact of temporal features vs weather features

### **Step 9: Prediction Pipeline**

Three prediction modes supported:

1. **Past Date (2023-2024):** Look up actual weather from dataset
2. **Tomorrow:** Fetch today's real weather (Open-Meteo API)
3. **Future (1-16 days):** Fetch forecast weather (Open-Meteo API)

---

## Running the Notebook

### **Prerequisites**

```bash
pip install pandas numpy scikit-learn matplotlib seaborn requests jupyter xgboost
```

### **Launch Jupyter**

```bash
jupyter notebook singapore_energy_forecast.ipynb
```

---

## Libraries & Tools

| Library          | Purpose                                |
| ---------------- | -------------------------------------- |
| **pandas**       | Data loading, cleaning, manipulation   |
| **numpy**        | Numerical operations                   |
| **scikit-learn** | ML models (LR, RF, metrics, scaling)   |
| **matplotlib**   | Static visualizations                  |
| **seaborn**      | Statistical plots (heatmaps, boxplots) |
| **requests**     | Open-Meteo API calls                   |
| **xgboost**      | Alternative gradient boosting model    |

---
