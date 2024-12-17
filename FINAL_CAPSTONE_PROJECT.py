#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


# In[2]:


import pandas as pd

# Load the energy dataset
energy_data = pd.read_csv('energy_dataset.csv')

# Load the weather dataset
weather_data = pd.read_csv('weather_features.csv')

# Display the first few rows to verify
print(energy_data.head())
print(weather_data.head())


# In[3]:


# Read datasets into pandas DataFrames
energy_data = df1 = pd.read_csv("energy_dataset.csv", parse_dates=['time'], index_col=['time'])
weather_data = df2 = pd.read_csv("weather_features.csv", parse_dates=['dt_iso'], index_col=['dt_iso'])


# In[4]:


# Data Cleaning and Preprocessing
# Handle timezone issues and align both datasets
df1.index = pd.to_datetime(df1.index, utc=True).tz_localize(None)
df2.index = pd.to_datetime(df2.index, utc=True).tz_localize(None)


# In[5]:


# Drop irrelevant or redundant columns
df1.drop(columns=['generation fossil coal-derived gas', 'generation fossil oil shale', 'generation marine'], inplace=True)
df2.drop(columns=['rain_3h', 'weather_icon'], inplace=True)


# In[6]:


# Check columns with all missing values
fully_missing_columns = df1.columns[df1.isnull().all()]
print("Columns with all missing values:", fully_missing_columns)

# Drop fully missing columns before imputation
df1_reduced = df1.drop(columns=fully_missing_columns)

# Apply IterativeImputer on the reduced DataFrame
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(df1_reduced)

# Create a DataFrame from the imputed data
df1_imputed = pd.DataFrame(imputed_data, columns=df1_reduced.columns, index=df1_reduced.index)

# Add back the fully missing columns (as NaN)
for col in fully_missing_columns:
    df1_imputed[col] = pd.NA

# Ensure the original column order is maintained
df1_imputed = df1_imputed[df1.columns]


# Boxplot of each column to look at the spread of data and check for outliers - no obvious outliers so read to use the df_energy dataframe. Looks like the biggest sources of energy are fossil gas, wind onshore, nuclear, hydro reservoir and solar 

# In[7]:


# Boxplots of each column with labels at 90 degrees for easy reading
df1.boxplot(rot=90, color="blue", figsize=(15, 5))

# Add a new column 'total generation' as the sum of all types of energy generated
df1['total generation'] = (
    df1['generation fossil gas'] +
    df1['generation fossil hard coal'] +
    df1['generation nuclear'] +
    df1['generation wind onshore'] +
    df1['generation other'] +
    df1['generation other renewable'] +
    df1['generation waste'] +
    df1['generation hydro water reservoir'] +
    df1['generation hydro pumped storage consumption'] +
    df1['generation hydro run-of-river and poundage'] +
    df1['generation solar']
)


# Plots of Generation type versus date - visualising the generation amounts by type

# In[8]:


import matplotlib.pyplot as plt

# Plot monthly mean generation data for each energy type
# Adjusted to use 'df1' (your merged dataframe)

fig, axes = plt.subplots(figsize=(25, 10))

# Plot total generation
df1['total generation'].resample('M').mean().plot(marker='.', alpha=0.8, label='Total Generation', ax=axes)

# Plot each generation type
df1['generation fossil gas'].resample('M').mean().plot(marker='.', alpha=0.8, label='Fossil Gas', ax=axes)
df1['generation fossil hard coal'].resample('M').mean().plot(marker='.', alpha=0.8, label='Fossil Hard Coal', ax=axes)
df1['generation nuclear'].resample('M').mean().plot(marker='.', alpha=0.8, label='Nuclear', ax=axes)
df1['generation wind onshore'].resample('M').mean().plot(marker='.', alpha=0.8, label='Wind Onshore', ax=axes)
df1['generation other'].resample('M').mean().plot(marker='.', alpha=0.8, label='Other', ax=axes)
df1['generation other renewable'].resample('M').mean().plot(marker='.', alpha=0.8, label='Other Renewable', ax=axes)
df1['generation waste'].resample('M').mean().plot(marker='.', alpha=0.8, label='Waste', ax=axes)
df1['generation hydro water reservoir'].resample('M').mean().plot(marker='.', alpha=0.8, label='Hydro Water Reservoir', ax=axes)
df1['generation hydro pumped storage consumption'].resample('M').mean().plot(marker='.', alpha=0.8, label='Hydro Pumped Storage Consumption', ax=axes)
df1['generation hydro run-of-river and poundage'].resample('M').mean().plot(marker='.', alpha=0.8, label='Hydro Run-of-River and Poundage', ax=axes)
df1['generation solar'].resample('M').mean().plot(marker='.', alpha=0.8, label='Solar', ax=axes)

# Customize plot labels and title
axes.legend(loc=(1.01, .01), ncol=1, fontsize=15)  # Adjust legend placement
axes.set_title('Monthly Mean Generation Amount by Energy Type', fontsize=30)
axes.set_ylabel('Monthly Mean Generation Amount (GMh)', fontsize=20)
axes.set_xlabel('Year', fontsize=20)

# Ensure layout is tight and readable
plt.tight_layout()
plt.show()


# A better way to visualise is by percentage of total energy generated by source resampled monthly (figures are provided for every hour of the day, but give too many data points to plot nicely so resampling by month is used here) Interestingly no single source provides more than 30% of the total (only twice are figures above 25% ) and the 5 sources that provide at least more than 10% are nuclear, fossil hard coal, onshore wind, fossil gas and hydro water reservoir.

# In[9]:


# Plot the percentage of total generation by type, for each energy source
fig, axes = plt.subplots(figsize=(25, 10))

# Fossil Gas Percentage
df1['generation fossil gas'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Fossil Gas', ax=axes)

# Fossil Hard Coal Percentage
df1['generation fossil hard coal'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Fossil Hard Coal', ax=axes)

# Nuclear Percentage
df1['generation nuclear'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Nuclear', ax=axes)

# Wind Onshore Percentage
df1['generation wind onshore'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Wind Onshore', ax=axes)

# Other Percentage
df1['generation other'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Other', ax=axes)

# Other Renewable Percentage
df1['generation other renewable'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Other Renewable', ax=axes)

# Waste Percentage
df1['generation waste'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Waste', ax=axes)

# Hydro Water Reservoir Percentage
df1['generation hydro water reservoir'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Hydro Water Reservoir', ax=axes)

# Hydro Pumped Storage Consumption Percentage
df1['generation hydro pumped storage consumption'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Hydro Pumped Storage Consumption', ax=axes)

# Hydro Run-of-River and Poundage Percentage
df1['generation hydro run-of-river and poundage'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Hydro Run-of-River and Poundage', ax=axes)

# Solar Percentage
df1['generation solar'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8, label='Solar', ax=axes)

# Customize plot labels and title
axes.legend(loc=(1.01, .01), ncol=1, fontsize=15)  # Adjust legend placement
axes.set_title('Percentage Generation by Type', fontsize=40)
axes.set_ylabel('Percentage (%)', fontsize=30)
axes.set_xlabel('Year', fontsize=20)

# Ensure layout is tight and readable
plt.tight_layout()
plt.show()


# Total percentages of production averaged over the 4 year period Some seasonality of production is evident too - solar for example peaks each July which make sense as that is summer and solar energy is stronger, the use of fossil hard coal dips each february/march (not sure why)

# In[10]:


# Calculate the percentages and print out.
print("Ordered percentage of total power generated over the 4 years by each source:")
print()

print("generation nuclear:", round((df1['generation nuclear'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation fossil gas:", round((df1['generation fossil gas'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation wind onshore:", round((df1['generation wind onshore'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation fossil hard coal:", round((df1['generation fossil hard coal'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation hydro water reservoir:", round((df1['generation hydro water reservoir'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation solar:", round((df1['generation solar'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation hydro run-of-river and poundage:", round((df1['generation hydro run-of-river and poundage'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation hydro pumped storage consumption:", round((df1['generation hydro pumped storage consumption'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation waste:", round((df1['generation waste'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation other renewable:", round((df1['generation other renewable'].sum() / df1['total generation'].sum()) * 100, 1), '%')
print("generation other:", round((df1['generation other'].sum() / df1['total generation'].sum()) * 100, 1), '%')


# Below plots show day ahead price vs actual (monthly means values) - the day ahead price is always cheaper (not always but sampled monthly it is - daily the graph is too messy) - so forecasting demand a day ahead accurately could be very useful - that is the task performed later.

# In[11]:


total_load = ['price actual','price day ahead']
group = df1[total_load].groupby(pd.Grouper(freq='M')).mean()
axes = group.plot(kind='line',marker='x', alpha=0.7, figsize=(15,6))
axes.set_ylabel('Cost $', fontsize=20)
# price the day ahead and the actual price averaged over the year
prices = ['price day ahead','price actual' ]
df1.groupby(df1.index.year)[prices].mean()


# Monthly average graph of total load forecast and actual, then monthly price forecast versus actual Averaged over a month the load forecast is very similar to the actual wich is a sign the forecasting is good on average over a month (the daily plot is too messy to include but it is needed for Q2). 

# In[12]:


# Load forecast vs actual - Monthly resampled
total_load = ['total load forecast', 'total load actual']
group = df1[total_load].groupby(pd.Grouper(freq='M')).mean()
axes = group.plot(marker='x', alpha=0.9, linestyle='None', figsize=(20,7))
axes.set_ylabel('Monthly Totals (GWh)', fontsize=20)
axes.set_title('Load Forecast vs Actual (Monthly)', fontsize=22)

# Load forecast vs actual - Daily resampled
group = df1[total_load].groupby(pd.Grouper(freq='D')).mean()
axes = group.plot(marker='x', alpha=0.9, linestyle='None', figsize=(20,7))
axes.set_ylabel('Daily Totals (GWh)', fontsize=20)
axes.set_title('Load Forecast vs Actual (Daily)', fontsize=22)


# Now some exploratory data analysis with the weather dataset. Including doing the same thing with datetime index as with energy dataset. Including removing the first value which become a 2014 value)

# In[13]:


# to show the parse dates hasn't worked so need to do explicitly in next step
print(df2.index.name)
print(df2.index.dtype)


# In[14]:


# make sure the index is in datetime format.
df2.index = pd.to_datetime(df2.index, utc='True')
#df2.index = df2.index.tz_convert('Europe/Madrid') - can't do this or get repeat index values
df2.index
# use tz_localize to remove the timezone but keep the time the same
df2 = df2.tz_localize(None)
# remove the 1 value that becomes a 2014 value
df2 = df2[df2.index.year >=2015]
df2.index
df2.index.dtype


# In[15]:


"""Checking for na values - none"""
df2.isna().sum()
df2.columns


# Drop the Kelvin temperatures and keep the celsius instead, Also Rain_1h is redundant given rain_1h, and full weather description and icon aren't need either

# In[16]:


# drop these columns - keep temperature columns in Celsius,
df2.drop(columns = [ 'rain_1h','weather_id','weather_description' ],axis=1,
inplace=True)
df2.shape


# In[17]:


# check for duplicates - yes
df2.index.has_duplicates


# In[18]:


# So need to drop any duplicates
df2 = df2.drop_duplicates(keep='first')


# From the above report there is no real data missing - just need to check number of records for each city - all relatively similar ~ 35,000, so no location bias is likely

# In[19]:


# As a check of number of records for each city simply check the number of temperature records and then plot.
df2.groupby(["city_name"])[['temp']].count().plot.bar(color="blue", legend=False,
figsize=(12,5))


# So the count of values from each site is very similar (from the above bar chart). Next check for outliers and/or incorrect values

# In[20]:


# boxplots for all features with labels rotated for readability. There seems to be a problem with pressure
df2.boxplot(rot='vertical', color='blue',figsize=(19, 5))


# There is definitely a problem with some pressure values. Pressure is generally measured in hectopascals hPa and actually has a tight range - definitely no less than 950 or higher that
# Here is a link to record values for Spanish max/min pressure - min is 950hPa, max is 1051hPa so will use these to cut off values. https://en.wikipedia.org/wiki/List_of_atmospheric_pressure_records_in_Europe#Spain

# In[21]:


# from the above there seems to be a problem with pressure
df2['pressure'].describe()


# In[22]:


# drop any duplicates
df2 = df2.drop_duplicates(keep='first')


# So lets look at the spread and count of values less than 955 or greater than 1051 (1228 with
# bad pressure values)

# In[23]:


sum(df2['pressure']<=955) + sum(df2['pressure']>=1051)
# examining some records shows that the pressure is just wrong - see the pressure column.
df2[df2['pressure']>=1051]


# Based on the fact there are 169774 rows of data I will simply remove the rows with pressure errors rather than try to correct as they are ony 1309 rows out of 169774 total or 0.01% of the total.

# In[24]:


# remove pressure values less than 955 and greater than 1055
df2 = df2[df2['pressure']>= 955]
df2 = df2[df2['pressure']<= 1051]


# In[25]:


# rows with bad pressure data removed
df2.shape


# In[26]:


# boxplots for all features with labels rotated for readability.Pressue values fixed - but now wind_speed seems to have a large value
df2.boxplot(rot='vertical', color="blue",figsize=(19, 5))
df2 = df2[df2['pressure']<=1051]
df2.boxplot(column='pressure', color="red",figsize=(5, 5))


# In[27]:


# checking the wind speed - max wind speed seems to be twice the next maximum and 130 km/h is a strong value
df2.boxplot(column='wind_speed', color="blue",figsize=(5, 5))


# Remove the windspeeds greater than 60km/h and redo boxplot - looks better now with the 134km/h and 64km/h values removed.

# In[28]:


df2 = df2[df2.wind_speed<=60]
df2.boxplot(column='wind_speed', color="blue",figsize=(5, 5))


# Last step is to convert temp, temp_min and temp_max from kelvin to celsius - simply subtract 273.15 and round to 1 decimal place and create new columns temp_C, temp_C_min, temp_C_max, then delete the temp, temp_max amd temp_min columns.

# In[29]:


# convert from Kelvin to Celcuis by subtracting 273.15 and create new columns temp_C, temp_C_min, temp_C_max
df2['temp_C'] = round((df2['temp']-273.15),1)
df2['temp_C_max'] = round((df2['temp_max']-273.15),1)
df2['temp_C_min'] = round((df2['temp_min']-273.15),1)
df2.drop(columns = ['temp','temp_max','temp_min'], axis=1, inplace=True)
# boxplots for all features with labels rotated for readability.
df2.boxplot(rot='vertical', color="blue",figsize=(19,5))
# can do similar plot for df_joined_pred
df2 = df2[df2.temp_C<=100]
df2.boxplot(column='temp_C', color="blue",figsize=(5, 5))


# Cities in the dataset are 'Valencia', 'Madrid', 'Bilbao', ' Barcelona', 'Seville Plotting the daily temperatures of each city. It shows the Seville (black) seems to have the highest temperatures and Bilbao (blue) and Madrid (red) have the lowest but the all follow the seasonal pattern.

# In[30]:


# Plots of monthly temps for all 3 cities - to make reasmpling daily change the resample("M") to D or W for weekly
axes = df2[df2.city_name=='Madrid' ]["temp_C"].resample("M").mean().plot(marker='.',
alpha=0.8, figsize=(25,10), label='Madrid', color='red' )
axes = df2[df2.city_name=='Bilbao' ]["temp_C"].resample("M").mean().plot(marker='x',
alpha=0.8, figsize=(25,10), label='Bilbao', color='blue')
axes = df2[df2.city_name=='Valencia' ]["temp_C"].resample("M").mean().plot(marker='*',
alpha=0.8, figsize=(25,10), label='Valencia', color='green')
axes = df2[df2.city_name=='Seville' ]["temp_C"].resample("M").mean().plot(marker='+',
alpha=0.8, figsize=(25,10), label='Seville', color='black')
axes = df2[df2.city_name==' Barcelona']["temp_C"].resample("M").mean().plot(marker='.',
alpha=0.8, figsize=(25,10), label='Barcelona', color='orange')
axes.legend(loc='upper left', frameon=False, fontsize=15)
axes.set_title('Daily temperatures for each location', fontsize=30)
axes.set_ylabel('Temperature (C)', fontsize=20)
axes.set_xlabel("Year", fontsize=20)


# Research Questions"
# 1. Which regression technique will accurately forecast the daily energy consumption demand using hourly period.
# 2. How to accurately forecast energy demand 24 hour in advance compared to TSO?
# 3. Using Classification, determine what weather measurements, and cities influence most the electrical demand, prices, generation capacity?

# In[31]:


# Visualise by total actual load by hour
import seaborn as sns
sns.boxplot(x=df1.index.hour, y='total load actual', data=df1,palette="twilight_shifted")
plt.ylabel('Total load (MW)', fontsize=24)
plt.xlabel('Hour', fontsize=24)
plt.title("Hourly boxplots of total load", fontsize=34)


# In[32]:


# Now to visualise load by month
sns.boxplot(x=df1.index.month_name(), y='total load actual', data=df1, palette="turbo")
plt.ylabel('Total load (MW) ', fontsize=24)
plt.xlabel('Month', fontsize=24)
plt.title("Monthly boxplots of total load Actual", fontsize=34)


# In[33]:


# Now to visualise load by day
sns.boxplot(x=df1.index.day_name(), y='total load actual', data=df1, palette="Set2")
plt.ylabel('Total load (MW) ', fontsize=24)
plt.xlabel('Day', fontsize=24)
plt.title("Daily boxplots of total load actual", fontsize=34)


# Since there are weather values for all 5 cities take the mean across all the cities - this will miss some extremes but is an easy way to summarise the weather df and then join with the energy df

# In[34]:


# Check columns of df1
print("Columns in df1:")
print(df1.columns)

# Check columns of df2
print("Columns in df2:")
print(df2.columns)

# Step 1: Convert 'Datetime' in df1 and 'dt_iso' in df2 to Date (ignoring time)
df1['Date'] = pd.to_datetime(df1['Datetime']).dt.date  # Extract date from 'Datetime' in df1
df2['Date'] = pd.to_datetime(df2['dt_iso']).dt.date    # Extract date from 'dt_iso' in df2

# Step 2: Merge df1 and df2 on the 'Date' column
df_joined = df2.merge(df1, on='Date', how='inner')  # Perform the merge on the 'Date' column

# Step 3: Check the shape and first few rows of the merged DataFrame
print(f"Shape of df_joined after merging: {df_joined.shape}")
print(df_joined.head())


# Creating a correlation matrix and displaying it shows a few interesting things. Namely the temperature only has a 20% correlation with the total load actual. Two graphs shown - one for all correlations and then one with absolute values for correlations greater than 40%, lastly the specific correlations can be viewed in a list (select the field manually).

# In[94]:


#Step 1: Calculate the correlation matrix for numeric columns
corr_matrix = df_joined.corr()

# Step 2: Display the correlation matrix
print("Correlation Matrix:")
print(corr_matrix)


# In[95]:


# Step 3: Visualize the correlation matrix using a heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Create a mask to display only the lower triangle of the matrix
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

# Set up the figure size
plt.figure(figsize=(16, 12))

# Create the heatmap
sns.heatmap(corr_matrix, annot=True, mask=mask, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Add a title and adjust layout
plt.title("Correlation Heatmap", fontsize=18)
plt.tight_layout()

# Show the plot
plt.show()


# To address your research questions, we can take a step-by-step approach and use various regression and classification techniques. Here's how we can proceed:
# 

# - Linear Regression (RLHE): A basic approach for regression problems. It could be used as a baseline model to see how well simple linear relationships between features (e.g., time of day, weather conditions) and energy demand perform.
# - Decision Tree Regressor (DTR): Decision Trees can model non-linear relationships in the data, making them more flexible than linear regression for capturing complex patterns in energy demand.
# - Random Forest Regressor (RFR): A more powerful ensemble method that aggregates multiple decision trees to improve prediction accuracy by reducing overfitting.
# - Artificial Neural Networks (ANN): Neural networks can capture complex patterns and interactions in the data, especially when there is a lot of non-linear relationships. This would be useful for accurately forecasting energy demand.

# Using Mean Squared Error (MSE) and R-squared (R²) to evaluate the model performance. Lower MSE and higher R² indicate better predictions.

# In[61]:


# Align datasets
energy_data.index = pd.to_datetime(energy_data.index, utc=True).tz_localize(None)
weather_data.index = pd.to_datetime(weather_data.index, utc=True).tz_localize(None)


# Handle missing values in energy data using IterativeImputer
fully_missing_columns = energy_data.columns[energy_data.isnull().all()]
print("Columns with all missing values:", fully_missing_columns)
energy_data_reduced = energy_data.drop(columns=fully_missing_columns)


# In[62]:


# Drop irrelevant columns
energy_data.drop(columns=['generation hydro pumped storage aggregated', 'forecast wind offshore eday ahead'], inplace=True)


# In[63]:


# Handle missing values in energy data using IterativeImputer
fully_missing_columns = energy_data.columns[energy_data.isnull().all()]
print("Columns with all missing values:", fully_missing_columns)
energy_data_reduced = energy_data.drop(columns=fully_missing_columns)


# In[64]:


# Apply IterativeImputer for imputation
imputer = IterativeImputer(max_iter=10, random_state=0)
imputed_data = imputer.fit_transform(energy_data_reduced)
energy_data_imputed = pd.DataFrame(imputed_data, columns=energy_data_reduced.columns, index=energy_data_reduced.index)


# In[65]:


# Re-add the fully missing columns with NaN values
for col in fully_missing_columns:
    energy_data_imputed[col] = pd.NA

# Ensure the original column order is maintained
energy_data_imputed = energy_data_imputed[energy_data.columns]


# In[66]:


# Create a new feature for total generation
energy_data_imputed['total generation'] = (
    energy_data_imputed['generation fossil gas'] +
    energy_data_imputed['generation fossil hard coal'] +
    energy_data_imputed['generation nuclear'] +
    energy_data_imputed['generation wind onshore'] +
    energy_data_imputed['generation other'] +
    energy_data_imputed['generation other renewable'] +
    energy_data_imputed['generation waste'] +
    energy_data_imputed['generation hydro water reservoir'] +
    energy_data_imputed['generation hydro pumped storage consumption'] +
    energy_data_imputed['generation hydro run-of-river and poundage'] +
    energy_data_imputed['generation solar']
)


# In[67]:


# Now, let's prepare X (features) and Y (target)
# Example: let's assume that the target variable is 'total generation', and we'll use weather features as inputs
X = weather_data.join(energy_data_imputed[['total generation']], how='inner').dropna()


# In[68]:


# Feature selection - drop the target from X (because it's the dependent variable)
y = X.pop('total generation')


# In[69]:


# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[88]:


# Display data types of the columns
print(y_train.dtypes)


# In[80]:


# Display data types of the columns
print(X_train.dtypes)


# In[81]:


from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize LabelEncoder for categorical columns
label_encoder = LabelEncoder()

# Encode the 'weather_main' and 'weather_description' columns
X_train['weather_main'] = label_encoder.fit_transform(X_train['weather_main'])
X_test['weather_main'] = label_encoder.transform(X_test['weather_main'])

X_train['weather_description'] = label_encoder.fit_transform(X_train['weather_description'])
X_test['weather_description'] = label_encoder.transform(X_test['weather_description'])

# Initialize StandardScaler for feature scaling
scaler = StandardScaler()

# Apply scaling to the entire dataset after encoding
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[82]:


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[83]:


# Regression Model 1: Linear Regression (RLHE)
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print(f"Linear Regression MSE: {lr_mse}")
print(f"Linear Regression R2: {lr_r2}")


# In[84]:


# Regression Model 2: Random Forest Regressor (RAG)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"Random Forest MSE: {rf_mse}")
print(f"Random Forest R2: {rf_r2}")


# In[85]:


# Regression Model 3: Artificial Neural Network (ANN)
ann_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
ann_model.fit(X_train_scaled, y_train)
ann_predictions = ann_model.predict(X_test_scaled)
ann_mse = mean_squared_error(y_test, ann_predictions)
ann_r2 = r2_score(y_test, ann_predictions)

print(f"ANN MSE: {ann_mse}")
print(f"ANN R2: {ann_r2}")


# # Based on the results provided:
# Linear Regression: MSE = 248,106,041,671.95, R² = 0.00015
# Random Forest: MSE = 261,757,262,200.23, R² = -0.05486
# Artificial Neural Network (ANN): MSE = 248,088,070,481.65, R² = 0.00022
# Analysis:
# 
# Linear Regression and ANN provide very similar performance, with slightly better R² for ANN (0.00022 vs. 0.00015).
# Random Forest has a worse performance with a negative R², indicating that it does not fit the data well in this case.
# Despite the low R² values (which indicate a poor fit to the data), ANN seems slightly better than Linear Regression for forecasting energy consumption. However, both methods perform poorly and may require further tuning or feature engineering.
# 

# In[ ]:




