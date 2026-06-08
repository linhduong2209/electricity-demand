
## 1. Data Collection

### 1.1 Fetch Open-Meteo Weather Data

Calls the Open-Meteo historical weather API for Singapore's coordinates, requesting daily variables.

| Variable                       | Description                                                                                                                                                                                                                                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`temperature_2m_max`**       | The **maximum** air temperature (in °C) measured at 2 meters above ground level during that day.                                                                                                                                                                                                                         |
| **`temperature_2m_min`**       | The **minimum** air temperature (in °C) measured at 2 meters above ground level during that day.                                                                                                                                                                                                                         |
| **`temperature_2m_mean`**      | The **average** air temperature (in °C) at 2 meters above ground level for the entire day.                                                                                                                                                                                                                               |
| **`relative_humidity_2m_max`** | The **highest** relative humidity percentage (%) recorded at 2 meters above ground during the day.                                                                                                                                                                                                                       |
| **`precipitation_sum`**        | The **total amount** of rain, snow, or other precipitation (in mm) that fell during the entire day.                                                                                                                                                                                                                      |
| **`wind_speed_10m_max`**       | The **maximum** wind speed (in km/h) measured at 10 meters above ground level during the day.                                                                                                                                                                                                                            |
| **`apparent_temperature_max`** | The **maximum** "feels-like" temperature (in °C) during the day.<br><br>- It is a calculated index that combines air temperature, relative humidity, and wind speed to determine how hot or cold it _feels_ to human skin.<br>- In Singapore, this is often higher than the actual temperature due to high humidity.<br> |
### 1.2 Load EMA Electricity Demand Data

https://www.ema.gov.sg/resources/statistics/half-hourly-system-demand-data

- Download Half-Hourly System Demand Data files from EMA website form 2023 to 2024. The raw data was obtained in Excel format, structured as weekly reports. 
- Reads a single weekly `.xls` file and extracts dates and sums the 48 half-hourly megawatt (MW) readings for each day, converting them to daily Megawatt-hours (MWh) by multiplying the sum by 0.5. 
- Combines the results into a single DataFrame, removes duplicate dates, sorts by date, and saves the final clean dataset to `ema_daily_demand.csv`.
### 1.3 Join Weather and Demand Data

- Merges the weather and demand DataFrames on the 'date' column.
- Ensures the date column is in datetime format and sorts the data.
## 2. Data Cleaning

Extracts calendar information from the date and flags weekends/public holidays as binary (0/1) features, because electricity demand behaves differently on those days.
## 3. Exploratory Data Analysis

Visualizes the data to understand relationships between weather and demand before building models. 
### 3.1 Time Series and Seasonal Patterns

Line plots for Daily Electricity Demand, Mean Temperature, and Maximum Humidity over the 2023–2024 period.

- **Demand Volatility:** Electricity demand shows significant daily fluctuation but follows a clear seasonal pattern.
- **Temperature Correlation:** There is a visible positive correlation between mean temperature and electricity demand. Peaks in temperature often align with peaks in energy consumption (likely due to air conditioning load).
- **Humidity:** Humidity remains relatively high and stable throughout the year, but spikes in humidity also appear to correlate with slight increases in demand.

![[Screenshot 2026-06-07 181951.png]]
### 3.2 Correlation Analysis & Weather-Demand Relationship

#### 3.2.1 Heatmap

Heatmap of the correlation matrix between all numerical features.
##### Strongest Positive Correlations with Demand

The most significant drivers of electricity demand are temperature-related variables:

- **`apparent_temp_max` (0.41):** This has the strongest positive correlation with demand. "Apparent temperature" accounts for humidity and wind chill, suggesting that how hot it _feels_ is a better predictor of energy usage (likely due to air conditioning load) than raw temperature alone.
- **`temp_mean` (0.53) & `temp_max` (0.44):** Mean and maximum temperatures also show strong positive correlations. As temperatures rise, demand increases significantly.
- **`temp_min` (0.38):** Minimum temperature has a moderate positive correlation, though weaker than mean or max.

##### Negative Correlations with Demand

- **`is_weekend` (-0.60):** This is the **strongest negative correlation** in the entire matrix. It indicates that electricity demand drops significantly on weekends compared to weekdays (likely due to reduced commercial and industrial activity).
- **`precipitation` (-0.10) & `wind_speed` (-0.19):** Rain and wind have weak negative correlations with demand. This might be because rainy/windy days are often cooler, reducing cooling needs, or because people stay indoors but engage in less energy-intensive activities.
- **`is_holiday` (-0.17):** Holidays also see a drop in demand, similar to weekends but slightly less pronounced.

##### Inter-variable Relationships

- **Temperature Variables:** `temp_max`, `temp_min`, and `temp_mean` are highly correlated with each other (0.74–0.75), which is expected.
- **Humidity & Temperature:** `humidity_max` has a negative correlation with `temp_max` (-0.56) and `temp_mean` (-0.33). This suggests that in this dataset, higher temperatures tend to coincide with lower maximum humidity (or vice versa), which is interesting for a tropical climate like Singapore where one might expect them to rise together.
- **Wind & Humidity:** `wind_speed` and `humidity_max` have a moderate negative correlation (-0.47).
#### Conclusion

Temperature features (especially `apparent_temp_max` and `temp_mean`) should be primary features in any predictive model for electricity demand.

#### 3.2.2 Scatter plot

Scatter plots of `temp_mean` vs. `demand_mwh` and `humidity_max` vs. `demand_mwh`.
##### Temperature vs. Demand

- **Strong Positive Correlation:** There is a clear, strong positive linear relationship between Mean Temperature and Electricity Demand. As the temperature rises from ~24°C to ~29°C, electricity demand consistently increases from ~135,000 MWh to over 175,000 MWh.
- **Linearity:** The data points form a distinct upward-sloping band, suggesting that a linear model would fit this relationship well. This aligns with the expectation that higher temperatures in Singapore drive up air conditioning usage, which is a major component of electricity consumption.
- **Variance:** While the trend is strong, there is still some variance (spread) at any given temperature, indicating that other factors (like day of the week or humidity) also influence demand.
##### Humidity vs. Demand

- **Weak/No Clear Linear Relationship:** Unlike temperature, Maximum Humidity does not show a clear linear relationship with electricity demand. The data points are scattered vertically across the entire range of humidity values (80% to 100%).
- **Vertical Clustering:** The plot shows distinct vertical lines of data points. This suggests that for any given humidity level, demand can vary significantly (from ~135k to ~175k MWh). This indicates that humidity alone is a poor predictor of demand compared to temperature.
- **Range Constraint:** The humidity data is heavily concentrated between 90% and 100%, which is typical for Singapore's tropical climate. This limited range might make it harder for a model to learn a strong relationship based solely on humidity.
### 3. 3 Demand by Day Type (Weekday vs Weekend vs Holiday)

Box plots comparing demand distributions by **Day of the Week** and **Holiday vs. Regular Day**.

#### Demand by Day of Week (Left Plot)

- **Weekdays (Mon–Fri):** Electricity demand is consistently high and stable across weekdays. The median demand hovers around **164,000 MWh**, with a relatively tight interquartile range (the box), indicating consistent industrial and commercial activity. Tuesday appears to have slightly higher peak demand than other weekdays. each weekday shows several outliers on the lower end (around 135,000–145,000 MWh). These likely correspond to public holidays that fell on a weekday during the dataset period.
- **Weekends (Sat–Sun):** There is a distinct drop in demand on weekends.
    - **Saturday:** Median demand drops to approximately **156,000 MWh**.
    - **Sunday:** Shows the lowest demand, with a median around **151,000 MWh**.
    - The spread (variability) is also slightly lower on weekends compared to weekdays.

#### Holiday vs. Regular Days (Right Plot)

- **Regular Days:** As seen in the left plot, regular days maintain a high baseline. The median is roughly **161,000 MWh**, with a wide range extending up to ~178,000 MWh.
- **Holidays:** Holidays show a significant reduction in electricity consumption.
    - The median demand drops to approximately **154,000 MWh**.
    - The distribution is shifted downwards significantly compared to regular days.
    - There are notable low-value outliers (around 135,000 MWh), suggesting some holidays result in exceptionally low grid usage (possibly major holidays like Chinese New Year or National Day where industrial shutdowns are common).

#### Conclusion

- **Workweek Effect:** The primary driver of demand variation is the workweek cycle. Weekends see a **5-7% reduction** in median demand compared to weekdays.
- **Holiday Impact:** Holidays behave similarly to weekends but can dip even lower, confirming that industrial/commercial load shedding is a major factor in Singapore's energy profile.
- **Consistency:** Weekday demand is remarkably consistent, suggesting a baseload driven by steady commercial/industrial operations, while the "noise" (outliers) comes from calendar events (holidays).
## 4. Feature Engineering
### 4.1 Create Lag and Rolling Features  

Lag features allow the model to learn from past values, capturing autocorrelation in electricity demand and temperature.

- `demand_lag_1`: Electricity demand from the previous day (t−1).
- `demand_lag_7`: Electricity demand from exactly one week ago (t−7)
- `temp_mean_lag_1`: Mean temperature from the previous day. 
- `temp_max_lag_1`: Maximum temperature from the previous day.

Rolling windows smooth out noise and help the model understand short-term and medium-term trends rather than just point-in-time values.

- `demand_rolling_7`: 7-day moving average of demand. Captures the weekly trend.
- `demand_rolling_30`: 30-day moving average of demand. Captures monthly trends/seasonality.
- `temp_mean_rolling_3`: 3-day rolling average of mean temperature. Smooths daily temperature fluctuations.
- `humidity_rolling_3`: 3-day rolling average of maximum humidity.

Interaction terms capture non-linear relationships between variables that individual features might miss. 

- `temp_humidity_interaction`: A multiplicative interaction between mean temperature and maximum humidity (`temp_mean * humidity_max / 100`). This approximates a "heat index" effect where high humidity makes high temperatures feel hotter, driving higher AC usage.

### 4.2 Prepare Data for Modeling

After engineering, the final dataset contains **20 features** used for modeling:

|Category|Features|
|---|---|
|**Raw Weather**|`temp_max`, `temp_min`, `temp_mean`, `humidity_max`, `precipitation`, `wind_speed`, `apparent_temp_max`|
|**Temporal**|`is_weekend`, `is_holiday`, `month`, `day_of_year`, `week_of_year`|
|**Lags**|`demand_lag_1`, `demand_lag_7`, `temp_mean_lag_1`|
|**Rolling**|`demand_rolling_7`, `demand_rolling_30`, `temp_mean_rolling_3`, `humidity_rolling_3`|
|**Interactions**|`temp_humidity_interaction`|
## 5. Train/Test Split & Baseline Forecast

### 5.1 Chronological Split (80% train, 20% test)

Because electricity demand is a time-series problem, the data is split chronologically rather than randomly. This prevents **data leakage** (using future data to predict the past) and ensures the model is evaluated on its ability to forecast future dates.

- **Total Feature Rows:** 723
- **Split Index:** 578 (80% of 723)
- **Training Set:** 578 samples (79.9%), covering the date range from **2023-01-09 to 2024-08-08**.
- **Test Set:** 145 samples (20.1%), covering the date range from **2024-08-09 to 2024-12-31**.
### 5.2 Establish Naive Baseline (Lag-1 Forecast)

A **Naive Baseline** is the simplest possible forecasting method. It relies on a single, intuitive assumption: **tomorrow will be exactly the same as today.**

This baseline serves as a fundamental benchmark. It establishes the minimum acceptable performance for any forecasting model. If a complex machine learning model cannot significantly outperform this naive "guess yesterday's value" approach, the machine learning model is not providing any real value. 
## 6. Model Training
### 6.1 Standardize Features

Before training, features are standardized using `StandardScaler` to ensure all variables have a mean of 0 and standard deviation of 1.
### 6.2 Regression Models (Exact MWh Prediction)

| Model                 | MAE (MWh) | RMSE (MWh) | R² Score |
| --------------------- | --------- | ---------- | -------- |
| **Baseline (Lag-1)**  | 4678.17   | 6367.63    | -0.1517  |
| **Linear Regression** | 5167.79   | 31979.38   | -28.0495 |
| **Random Forest**     | 1614.50   | 2155.33    | 0.8680   |
- **MAE (Mean Absolute Error):** It calculates the absolute difference between the predicted and actual values, then averages them. It treats all errors equally.
- **RMAE(Root Mean Squared Error):** It squares the errors before averaging them, and then takes the square root. Because it squares the errors, it **heavily penalizes large mistakes** (outliers).
- **R² Score (Coefficient of Determination:** It measures the proportion of variance in the target variable that the model explains. ( **1.0** = Perfect prediction, **0.0** = Your model is no better than just guessing the historical average every day, **< 0.0** = Your model is actually **worse** than just guessing the average)

**Conclusion:**

Best regression model: Random Forest
Improvement over baseline (MAE): 3063.67 MWh (65.5%)
### 6.3 Classification Models (High/Medium/Low Demand)

- **Low (0):** Demand ≤ 33rd percentile
- **Medium (1):** 33rd percentile < Demand ≤ 67th percentile
- **High (2):** Demand > 67th percentile

|Model|Accuracy|Macro F1 Score|
|---|---|---|
|**Logistic Regression**|**76.55%**|**0.7768**|
|**Random Forest**|74.48%|0.7621|

**Conclusion:** 

Best classification model: Logistic Regression
## 7. Model Evaluation

### 7.1 Regression Model Predictions vs Actual

- Actual demand range: 148627 - 175546 MWh
- Predicted range: 148436 - 172082 MWh
### 7.2 Classification Confusion Matrix & Metrics

- Accuracy: 0.7448
- Macro F1 Score: 0.7621