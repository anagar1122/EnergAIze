#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages

import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
from posixpath import join
import pandas as pd
from pandas.core import describe
import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas as pd
import numpy as np
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import operator
from sklearn.preprocessing import PolynomialFeatures


# In[2]:


from sklearn.decomposition import PCA as sklearnPCA
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2


# In[3]:


get_ipython().system('{sys.executable} -m pip install -U pandas-profiling')
get_ipython().system('jupyter nbextension enable --py widgetsnbextension')
get_ipython().system('pip install matplotlib')
get_ipython().system('pip install graphviz')
get_ipython().system('pip install sktime')


# In[4]:


# sktime libraries for time series forecasting

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.utils.plotting import plot_series


# In[5]:


# for imputing missing na values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# In[6]:


# set up a scaler
from sklearn.preprocessing import MinMaxScaler


# In[7]:


# Commented out IPython magic to ensure Python compatibility.
# to get the plots to show in the same size
# %matplotlib inline
plt.rcParams['figure.figsize'] = (25, 10)


# """Import the energy dataset and weather datasets. As with the energy dataframe need to set the time to UTC to remove the issues that arise with the move into/out of daylight savings - as it is based on Madrid timezone it is only 1 hour difference to UTC."""
# 
# 

# In[8]:


#upload data set


# In[9]:


df1 = pd.read_csv('energy_dataset.csv',parse_dates=['time'],index_col=['time'])

df2 = pd.read_csv('weather_features.csv',parse_dates=['dt_iso'],index_col=['dt_iso'])


# The data is in the Europe/Mardid timezone which has daylight savings. This in turn makes for repeated index values at the end of each daylight savings season. To avoid this the time field needs to be converted to remove this, using UTC format will do this and allow to set the index as datetime. There is only 1 hour timezone difference so insignifcant for analysis purposes. Also remove the first value which after conversion becomes the last hour in 2014 - hence select all values form 2015 or greater.

# In[10]:


# set the index as datetime type and set to UTC time - then set to correct timezone Europe/Madrid
df1.index = pd.to_datetime(df1.index, utc='True')
#df1.index = df1.index.tz_convert('Europe/Madrid')  -- can't be done or repeated values occur
# check datatype - now is a datetime
df1.index

# use tz_localize to remove the timezone but keep the time the same
df1 = df1.tz_localize(None)
# remove the 2014 value
df1 = df1[df1.index.year >=2015]
df1.index


# In[11]:


pip install ydata-profiling


# In[12]:


pip install numba==0.56.4


# In[13]:


# project: Predicting energy demand in Spain (four years of hourly unit electric consumption, generation, pricing, and weather data in Spain (5 cities))
# Use ydata_profiling library to investigate the data
from ydata_profiling import ProfileReport

profile1 = ProfileReport(df1, title="Pandas Profiling Report for df1", explorative=True)
profile1.to_notebook_iframe()


# Now to clean up the datset based on the report above. Several columns can be dropped as they contain no data or are all zero. No other columns have more than 0.1% of data missing so no need to perform any further cleaning/dropping/imputation - from pandas_profile

# Empty columns are: generation fossil coal-derived gas, generation fossil oil shale, generation fossil peat, generation geothermal, generation hydro pumped storage aggregated, generation marine, generation wind offshore, forecast wind offshore eday ahead ( My choosen dependent variable for classification which is not correct).

# In[14]:


# drop columns
df1 = df1.drop(columns=['generation fossil coal-derived gas', 'generation fossil oil shale', 'generation fossil peat', 'generation geothermal',
                        'generation hydro pumped storage aggregated', 'generation marine', 'generation wind offshore', 'forecast wind offshore eday ahead'],axis=1)


# In[15]:


# checking columns wered dropped - yes 8 columns dropped
df1.shape


# In[16]:


# check for any duplicates - none so no need to drop any duplicates
df1.index.has_duplicates


# Check for na values, and if there are any use a Iterative imputer from scikit learn to replace (it is a better estimator than using the mean or median like the SimpleImputer)

# In[17]:


df1.isna().sum()

# now to useIterative imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputer.fit(df1)
df1_filled = pd.DataFrame(imputer.transform(df1), columns = df1.columns, index=df1.index)

df1_filled.index

# check all na values have been imputed and there are no na's in the dtaframe
df1_filled.isna().sum()

# rename it back to df1
df1 = df1_filled


# In[18]:


# check shape again
df1.shape


# Boxplot of each column to look at the spread of data and check for outliers - no obvious outliers so read to use the df_energy dataframe.
# Looks like the biggest sources of energy are fossil gas, wind onshore, nuclear, hydro reservoir and solar
# 
# 

# In[19]:


# boxplots of each column with labels at 90 for eas of reading
df1.boxplot(rot='vertical', color="blue",figsize=(15,5))


# Add another column to 'total generation' as sum of all the types of energy generated.

# In[20]:


df1['total generation'] = df1['generation fossil gas']+df1['generation fossil hard coal']+df1['generation nuclear']+df1['generation wind onshore']+ df1['generation other']+df1['generation other renewable']+df1['generation waste']+df1['generation hydro water reservoir']+ df1['generation hydro pumped storage consumption']+df1['generation hydro run-of-river and poundage']+df1['generation solar']


# Plots of Generation type versus date - visualising the generation amounts by type

# In[21]:


#axes = df1['total generation'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='total')
axes = df1['generation fossil gas'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='fossil gas')
axes = df1['generation fossil hard coal'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='fossil hard coal')
axes = df1['generation nuclear'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='nuclear')
axes = df1['generation wind onshore'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='wind onshore')


axes = df1['generation other'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='other')
axes = df1['generation other renewable'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='other renewable')
axes = df1['generation waste'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='waste')
axes = df1['generation hydro water reservoir'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='generation hydro water reservoir')
axes = df1['generation hydro pumped storage consumption'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='hydro pumped storage consumption')
axes = df1['generation hydro run-of-river and poundage'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='hydro run-of-river and poundage')
axes = df1['generation solar'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='solar')

axes.legend(loc='upper right', frameon=False, fontsize=15)
axes.set_title('Generation Amount by Type', fontsize=30)
axes.set_ylabel('Monthly mean Generation Amount (GMh)', fontsize=20)
axes.set_xlabel("Year", fontsize=20)
axes.legend(loc=(1.01, .01), ncol=1, fontsize=15)
plt.tight_layout()


# A better way to visualise is by percentage of total energy generated by source resampled monthly (figures are provided for every hour of the day, but give too many data points to plot nicely so resampling by month is used here)
# 
# Interestingly no single source provides more than 30% of the total (only twice are figures above 25% ) and the 5 sources that provide at least more than 10% are nuclear, fossil hard coal, onshore wind, fossil gas and hydro water reservoir.

# In[22]:


axes = df1['generation fossil gas'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='fossil gas')
axes = df1['generation fossil hard coal'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='fossil hard coal')
axes = df1['generation nuclear'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='nuclear')
axes = df1['generation wind onshore'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='wind onshore')


axes = df1['generation other'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='other')
axes = df1['generation other renewable'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='other renewable')
axes = df1['generation waste'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='waste')
axes = df1['generation hydro water reservoir'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='generation hydro water reservoir')
axes = df1['generation hydro pumped storage consumption'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='hydro pumped storage consumption')
axes = df1['generation hydro run-of-river and poundage'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='hydro run-of-river and poundage')
axes = df1['generation solar'].resample("M").mean().div(df1['total generation'].resample("M").mean()).multiply(100).plot(marker='x', alpha=0.8,  figsize=(25,10), label='solar')

axes.legend(loc='upper right', frameon=False, fontsize=15)
axes.set_title('Percentage Generation by Type', fontsize=40)
axes.set_ylabel('Percentage', fontsize=30)
axes.set_xlabel("Year", fontsize=20)
axes.legend(loc=(1.01, .01), ncol=1, fontsize=20)
plt.tight_layout()


# Total percentages of production averaged over the 4 year period
# 
# 
# Some seasonality of production is evident too - solar for example peaks each July which make sense as that is summer and solar energy is stronger, the use of fossil hard coal dips each february/march (not sure why)

# In[23]:


# Calculate the percentages and print out.
print('Ordered percentage of total power generated over the 4 years by each source ')
print()
print("generation nuclear                         ",round((df1['generation nuclear'].sum()/df1['total generation'].sum())*100,1),'%')
print("generation fossil gas                      ",round((df1['generation fossil gas'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation wind onshore                    ',round((df1['generation wind onshore'].sum()/df1['total generation'].sum())*100,1),'%')
print("generation fossil hard coal                ",round((df1['generation fossil hard coal'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation hydro water reservoir            ',round((df1['generation hydro water reservoir'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation solar                            ',round((df1['generation solar'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation hydro run-of-river and poundage  ',round((df1['generation hydro run-of-river and poundage'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation hydro pumped storage consumption ',round((df1['generation hydro pumped storage consumption'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation waste                            ',round((df1['generation waste'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation other renewable                  ',round((df1['generation other renewable'].sum()/df1['total generation'].sum())*100,1),'%')
print('generation other                            ',round((df1['generation other'].sum()/df1['total generation'].sum())*100,1),'%')


# Below plots show day ahead price vs actual (monthly means values) - the day ahead price is always cheaper (not always but sampled monthly it is - daily the graph is too messy) - so forecasting demand a day ahead accurately could be very useful - that is the task performed later.

# In[24]:


total_load = ['price actual','price day ahead']

group = df1[total_load].groupby(pd.Grouper(freq='M')).mean()
axes = group.plot(kind='line',marker='x', alpha=0.7,  figsize=(15,6))
axes.set_ylabel('Cost $', fontsize=20)

# price the day ahead and the actual price averaged over the year
prices = ['price day ahead','price actual' ]
df1.groupby(df1.index.year)[prices].mean()


# Monthly average graph of total load forecast and actual, then monthly price forecast versus actual
# 
# Averaged over a month the load forecast is very similar to the actual wich is a sign the forecasting is good on average over a month (the daily plot is too messy to include but it is needed for Q2).

# In[25]:


# load forecast vs actual - monthly resampled
total_load = ['total load forecast','total load actual']

group = df1[total_load].groupby(pd.Grouper(freq='M')).mean()
axes = group.plot(marker='x', alpha=0.9, linestyle='None', figsize=(20,7))
axes.set_ylabel('Daily Totals (GWh)', fontsize=20)
# can do similar for 4 reg model

# load forecast vs actual - daily resampled
total_load = ['total load forecast','total load actual']

group = df1[total_load].groupby(pd.Grouper(freq='D')).mean()
axes = group.plot(marker='x', alpha=0.9, linestyle='None', figsize=(20,7))
axes.set_ylabel('Daily Totals (GWh)', fontsize=20)


# Now some exploratory data analysis with the weather dataset. Including doing the same thing with datetime index as with energy dataset. Including removing the first value which become a 2014 value)

# In[26]:


# to show the parse dates hasn't worked so need to do explicitly in next step
print(df2.index.name)
print(df2.index.dtype)


# In[27]:


# make sure the index is in datetime format.
df2.index = pd.to_datetime(df2.index, utc='True')
#df2.index = df2.index.tz_convert('Europe/Madrid') - can't do this or get repeat index values
df2.index


# In[28]:


# use tz_localize to remove the timezone but keep the time the same
df2 = df2.tz_localize(None)
# remove the 1 value that becomes a 2014 value
df2 = df2[df2.index.year >=2015]
df2.index

df2.index.dtype


# In[29]:


"""Checking for na values - none"""

df2.isna().sum()

df2.columns


# Drop the Kelvin temperatures and keep the celsius instead, Also Rain_3h is redundant given rain_1h, and full weather description and icon aren't need either

# In[30]:


# drop these columns - keep temperature columns in Celsius,
df2.drop(columns = [ 'rain_3h','weather_id','weather_description', 'weather_icon' ],axis=1, inplace=True)

df2.shape


# In[31]:


# check for duplicates - yes
df2.index.has_duplicates


# In[32]:


# So need to drop any duplicates
df2 = df2.drop_duplicates(keep='first')


# In[33]:


# check to shape to see if any duplicates were dropped - yes
df2.shape


# In[34]:


profile2 = ProfileReport(df2)
profile2.to_notebook_iframe()


# From the above report there is no real data missing - just need to check number of records for each city - all relatively similar ~ 35,000, so no location bias is likely

# In[35]:


# As a check of number of records for each city simply check the number of temperature records and then plot.
df2.groupby(["city_name"])[['temp']].count().plot.bar(color="blue", legend=False, figsize=(12,5))


# So the count of values from each site is very similar (from the above bar chart). Next check for outliers and/or incorrect values

# In[36]:


# boxplots for all features with labels rotated for readability. There seems to be a problem with pressure
df2.boxplot(rot='vertical', color='blue',figsize=(19, 5))


# There is definitely a problem with some pressure values. Pressure is generally measured in hectopascals hPa and actually has a tight range - definitely no less than 950 or higher that 1050. Here is a link to record values for Spanish max/min pressure - min is 950hPa, max is 1051hPa so will use these to cut off values. https://en.wikipedia.org/wiki/List_of_atmospheric_pressure_records_in_Europe#Spain

# In[37]:


# from the above there seems to be a problem with pressure
df2['pressure'].describe()


# In[38]:


# drop any duplicates
df2 = df2.drop_duplicates(keep='first')


# So lets look at the spread and count of values less than 955 or greater than 1051 (1228 with bad pressure values)

# In[39]:


sum(df2['pressure']<=955) + sum(df2['pressure']>=1051)


# In[40]:


# examining some records shows that the pressure is just wrong - see the pressure column.
df2[df2['pressure']>=1051]


# Based on the fact there are 169774 rows of data I will simply remove the rows with pressure errors rather than try to correct as they are ony 1309 rows out of 169774 total or 0.01% of the total.

# In[41]:


# remove pressure values less than 955 and greater than 1055
df2 = df2[df2['pressure']>= 955]
df2 = df2[df2['pressure']<= 1051]


# In[42]:


# rows with bad pressure data removed
df2.shape


# In[43]:


# boxplots for all features with labels rotated for readability.Pressue values fixed - but now wind_speed seems to have a large value
df2.boxplot(rot='vertical', color="blue",figsize=(19, 5))

df2 = df2[df2['pressure']<=1051]
df2.boxplot(column='pressure', color="red",figsize=(5, 5))


# In[44]:


# checking the wind speed - max wind speed seems to be twice the next maximum and 130 km/h is a strong value
df2.boxplot(column='wind_speed', color="blue",figsize=(5, 5))


# Remove the windspeeds greater than 60km/h and redo boxplot - looks better now with the 134km/h and 64km/h values removed.

# In[45]:


df2 = df2[df2.wind_speed<=60]
df2.boxplot(column='wind_speed', color="blue",figsize=(5, 5))


# Last step is to convert temp, temp_min and temp_max from kelvin to celsius - simply subtract 273.15 and round to 1 decimal place and create new columns temp_C, temp_C_min, temp_C_max, then delete the temp, temp_max amd temp_min columns.

# In[46]:


# convert from Kelvin to Celcuis by subtracting 273.15 and create new columns temp_C, temp_C_min, temp_C_max
df2['temp_C'] = round((df2['temp']-273.15),1)
df2['temp_C_max'] = round((df2['temp_max']-273.15),1)
df2['temp_C_min'] = round((df2['temp_min']-273.15),1)

df2.drop(columns = ['temp','temp_max','temp_min'], axis=1, inplace=True)


# In[47]:


# boxplots for all features with labels rotated for readability.
df2.boxplot(rot='vertical', color="blue",figsize=(19,5))


# In[48]:


# can do similar plot for df_joined_pred

df2 = df2[df2.temp_C<=100]
df2.boxplot(column='temp_C', color="blue",figsize=(5, 5))


# Cities in the dataset are 'Valencia', 'Madrid', 'Bilbao', ' Barcelona', 'Seville
# 
# Plotting the daily temperatures of each city. It shows the Seville (black) seems to have the highest temperatures and Bilbao (blue) and Madrid (red) have the lowest but the all follow the seasonal pattern.

# In[49]:


# Plots of monthly temps for all 3 cities - to make reasmpling daily change the resample("M") to D or W for weekly
axes = df2[df2.city_name=='Madrid'    ]["temp_C"].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='Madrid', color='red' )
axes = df2[df2.city_name=='Bilbao'    ]["temp_C"].resample("M").mean().plot(marker='x', alpha=0.8,  figsize=(25,10), label='Bilbao', color='blue')
axes = df2[df2.city_name=='Valencia'  ]["temp_C"].resample("M").mean().plot(marker='*', alpha=0.8,  figsize=(25,10), label='Valencia', color='green')
axes = df2[df2.city_name=='Seville'   ]["temp_C"].resample("M").mean().plot(marker='+', alpha=0.8,  figsize=(25,10), label='Seville', color='black')
axes = df2[df2.city_name==' Barcelona']["temp_C"].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='Barcelona', color='orange')
axes.legend(loc='upper left', frameon=False, fontsize=15)
axes.set_title('Daily temperatures for each location', fontsize=30)
axes.set_ylabel('Temperature  (C)', fontsize=20)
axes.set_xlabel("Year", fontsize=20)


# # Research Questions
# 1. Which regression technique will accurately forecast the daily energy consumption demand using hourly period.
# 
# 2. How to accurately forecast energy demand 24 hour in advance compared to TSO?
# 
# 3. Using Classification, determine what weather measurements, and cities influence most the electrical demand, prices, generation capacity?

# # --> Q1
# Which regression technique will accurately forecast the daily energy consumption demand using hourly period.
# 
# 

# In[50]:


# Visualise by total actual load by hour

sns.boxplot(x=df1.index.hour, y='total load actual', data=df1,palette="twilight_shifted")
plt.ylabel('Total load (MW)', fontsize=24)
plt.xlabel('Hour', fontsize=24)
plt.title("Hourly boxplots of total load", fontsize=34)


# In[51]:


# Now to visualise load by month

sns.boxplot(x=df1.index.month_name(), y='total load actual', data=df1, palette="turbo")
plt.ylabel('Total load (MW) ', fontsize=24)
plt.xlabel('Month', fontsize=24)
plt.title("Monthly boxplots of total load Actual", fontsize=34)


# In[52]:


# Now to visualise load by day

sns.boxplot(x=df1.index.day_name(), y='total load actual', data=df1, palette="Set2")
plt.ylabel('Total load (MW) ', fontsize=24)
plt.xlabel('Day', fontsize=24)
plt.title("Daily boxplots of total load actual", fontsize=34)


# Since there are weather values for all 5 cities take the mean across all the cities - this will miss some extremes but is an easy way to summarise the weather df and then join with the energy df

# In[53]:


# Create a combined df of the information from the 5 cities by taking the mean of the values for each hour.
df2_combined = df2.resample("H").mean()
df2_combined.index.rename('Datetime', inplace=True)

df2_combined.head()

df2_combined.info()

df2_combined.shape

df1.index.rename('Datetime', inplace=True)
df1.index.name


# In[54]:


# Next join the 2 datasets on the common index
df_joined = df2_combined.merge(df1, on='Datetime', how = 'inner')

display(df_joined.head())

df_joined.index

df_joined.info()

df_joined.isna().sum()

df_joined.dropna(inplace=True)

df_joined.isna().sum()



# In[56]:


# check for duplicates -
df_joined.index.has_duplicates


# In[57]:


df_joined.shape


# Creating a correlation matrix and displaying it shows a few interesting things. Namely the temperature only has a 20% correlation with the total load actual. Two graphs shown - one for all correlations and then one with absolute values for correlations greater than 40%, lastly the specific correlations can be viewed in a list (select the field manually).

# In[59]:



# To find the correlation among the columns using pearson method

corr_matrix = df_joined.corr().round(2) # add in .abs() if not worried about positive or negative correlation just the strength
corr_matrix


# In[60]:


# use this to show only lower triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cut_off = -1 # to show all correlations

mask |= np.abs(corr_matrix) < cut_off
corr = corr_matrix[~mask]  # fill in NaN in the non-desired cells
# remove empty rows/columns to make it easier to read.
remove_empty_rows_and_cols = True
if remove_empty_rows_and_cols:
    wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
    wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
    corr = corr.iloc[wanted_cols, wanted_rows]


# In[64]:


# display the correlation value in a grid to make it easier to read.
s = sns.heatmap(corr,annot=True,  linewidths=0.1, linecolor='gray')
# increase the size of the labels.
s.set_title('Correlation Heatmap (all correlations)', fontsize=40)
s.set_xticklabels(s.get_xmajorticklabels(), fontsize = 15)
s.set_yticklabels(s.get_ymajorticklabels(), fontsize = 15)
plt.show()


# In[66]:


# can do it for again after feature creation

# To find the correlation among the columns using pearson method and then only display values with correlations > 40%

corr_matrix = df_joined.corr().abs().round(2) # add in .abs() if not worried about positive or negative correlation just the strength
corr_matrix


# In[67]:


# use this to show only lower triangle
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cut_off = 0.40 # just show correlation greater than 40%

mask |= np.abs(corr_matrix) < cut_off
corr = corr_matrix[~mask]  # fill in NaN in the non-desired cells
# reove empty rows/columns to make it easier to read.
remove_empty_rows_and_cols = True
if remove_empty_rows_and_cols:
    wanted_cols = np.flatnonzero(np.count_nonzero(~mask, axis=1))
    wanted_rows = np.flatnonzero(np.count_nonzero(~mask, axis=0))
    corr = corr.iloc[wanted_cols, wanted_rows]
# display the correlation value in a grid to make it easier to read.
s = sns.heatmap(corr,annot=True,  linewidths=0.1, linecolor='gray')
# increase the size of the labels.
s.set_title('Correlation Heatmap (absolute value greater than 40%)', fontsize=40)
s.set_xticklabels(s.get_xmajorticklabels(), fontsize = 15)
s.set_yticklabels(s.get_ymajorticklabels(), fontsize = 15)
plt.show()


# Another way of viewing the correlations - this list can be changed to see different specific correlations - just change the specified field in the last line.

# In[68]:


# way of viewing specific correlations
correlations = df_joined.corr().unstack().sort_values(ascending=False).drop_duplicates().round(2)
correlations['total load forecast']  # change the required field here to get the specific correlations

correlations['humidity']  # change the required field here to get the specific correlations

correlations['wind_speed']

correlations['temp_C']

correlations['price actual']

correlations['price day ahead']

correlations['total generation']


# Since we are trying to predict the total load actual value, the values correlated to that need to be included as predictors - from below we can see that the values to use are :
# 
# ['total generation','generation fossil gas','generation fossil oil','generation hydro water reservoir','price actual','generation fossil hard coal','temp_C_max','generation other renewable','generation fossil brown coal/lignite,'forecast solar day ahead','wind_speed','humidity','generation hydro run-of-river and poundage','rain_1h', 'generation nuclear', 'generation biomass']

# In[69]:


correlations['total load actual'] # dependent variable


# # So the target variable is the total load actual and the predictors are as above

# In[70]:


display(df_joined.describe())


# In[71]:


#Check the data types of the attributes.
from numpy.lib import type_check
print(df_joined.dtypes)


# Now to see if a multivariate model can be trained using this data
# 
# - first make a smaller dataframe with the variables needed, then set up y_train/test and X_train/test

# In[72]:


# kind of Baseline Model
# I changed the dependent variable from total load forecast to total load actual
# create a new df; df_joined_pred - for prediction - only the correlated columns
df_joined_pred = df_joined[['total load actual','total generation','generation fossil gas','generation fossil oil','generation hydro water reservoir','price actual','generation fossil hard coal','temp_C_max','generation other renewable','generation fossil brown coal/lignite','forecast solar day ahead','wind_speed','humidity','generation hydro run-of-river and poundage','rain_1h', 'generation nuclear', 'generation biomass']]


# In[73]:


df_joined_pred.head()


# In[74]:


df_joined_pred.shape


# In[75]:


df_joined_pred
labels = np.array(df_joined_pred['total load actual'])
df_joined_pred_new = df_joined_pred.drop('total load actual', axis = 1)
df_joined_pred_new


# In[76]:


train_instances, test_instances, train_labels, test_labels = train_test_split (df_joined_pred_new, labels, test_size = 0.20, random_state = 4180)


# In[77]:


print('Training instances shape:', train_instances.shape)
print('Training labels shape:', train_labels.shape)
print('Testing instances shape:', test_instances.shape)
print('Testing labels shape:', test_labels.shape)


# In[88]:


from sklearn.metrics import mean_squared_error, r2_score
import math

# Function to evaluate mean squared error, RMSE, and R2 score
def evaluate_results(TestLabel, TestPredictions):
    mse = round(mean_squared_error(TestLabel, TestPredictions), 2)
    rmse = round(math.sqrt(mse), 2)  # Ensure rmse is rounded
    r2_score_value = round(r2_score(TestLabel, TestPredictions), 2)

    print("MSE: " + str(mse))
    print("RMSE: " + str(rmse))
    print("R2 Score: " + str(r2_score_value))


# In[92]:


# Linear Regression
regr = linear_model.LinearRegression()
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
test_predictions


# In[91]:


train_labels

test_labels


# My understanding is that test_prediction is taking training model and predict test_label. then error is calculated by find difference between test_label and test_prediction.

# In[93]:


# Linear Regression Continued
evaluate_results(test_labels, test_predictions)


# In[94]:


# can try on toal load forecast and total load actual


# In[95]:


# Python Function to return highest r2_score after consecutively training multiple KNN regression models
def evaluate_results2(TestLabel, TestPredictions):
    mse_2 = round(mean_squared_error(TestLabel, TestPredictions), 2)
    rmse_2 = math.sqrt(mse_2)
    r2_score_value_2 = round(r2_score(TestLabel, TestPredictions), 2)

    return r2_score_value_2


n = 5
max_r2_score = 0

while n <= 10:
    regr = KNeighborsRegressor (n_neighbors = n) #number of neighbors is arbitrary
    model = regr.fit(train_instances, train_labels)

    test_predictions = model.predict(test_instances)
    r2_score_value_2 = evaluate_results2(test_labels, test_predictions)

    if r2_score_value_2 >= max_r2_score:
        max_r2_score = r2_score_value_2

    n = n + 1

print ("Maximum r2_score_value " + str (max_r2_score))


# In[96]:


# KNN Regression
regr = KNeighborsRegressor (n_neighbors = 5) #number of neighbors is arbitrary
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)


# In[97]:


# Decision Tree Regression
regr = DecisionTreeRegressor(random_state = 42)
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)


# In[98]:


# Random Forest Regression
regr = RandomForestRegressor(random_state = 42, n_estimators = 10)
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)


# # --> Q2 How to accurately forecast energy demand 24 hour in advance compared to TSO?

# In[99]:


# For the hourly data
print('The mean absolute percentage error = {:.2f}%'.format(mean_absolute_percentage_error(df1['total load forecast'],df1['total load actual'])*100))


# In[100]:


# resampled daily
print('The mean absolute percentage error rsampled daily = {:.2f}%'.format(mean_absolute_percentage_error(df1['total load forecast'].resample('D').mean(),df1['total load actual'].resample('D').mean())*100))


# In[101]:


print('The RMSE error = {:.2f}%'.format(mean_squared_error(df1['total load actual'].resample('D').mean(),df1['total load forecast'].resample('D').mean(), squared=False)))
# or 3.09 which is way better than RF regressor which has rmse of 1155


# So the number that needs to be beaten to show we can forecast better than the TSO is a MAPE < 0.0082 or RMSE < 3.09
# 

# In[102]:


#Feature Creation

def create_features(df_joined_pred_new):
  

  df_joined_pred_new = df_joined_pred_new.copy()
  df_joined_pred_new['hour'] = df_joined_pred_new.index.hour
  df_joined_pred_new['dayofweek'] = df_joined_pred_new.index.day_of_week # monday is zero and sunday is 6
  df_joined_pred_new['quarter'] = df_joined_pred_new.index.quarter
  df_joined_pred_new['month'] = df_joined_pred_new.index.month
  df_joined_pred_new['year'] = df_joined_pred_new.index.year
  df_joined_pred_new['dayofyear'] = df_joined_pred_new.index.dayofyear
  return df_joined_pred_new

df_joined_pred_new = create_features(df_joined_pred_new)

train_instances = create_features(train_instances)
test_instances = create_features(test_instances)

train_instances, test_instances, train_labels, test_labels = train_test_split (df_joined_pred_new, labels, test_size = 0.20, random_state = 4180)

print('Training instances shape:', train_instances.shape)
print('Training labels shape:', train_labels.shape)
print('Testing instances shape:', test_instances.shape)
print('Testing labels shape:', test_labels.shape)

df_joined_pred_new.head()


# In[103]:


# Linear Regression
regr = linear_model.LinearRegression()
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)


# creation time features improves the accuracy slightly

# In[104]:


def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    return np.mean(np.abs((actual - pred) / actual)) * 100

df_daily = df_joined_pred.resample("D").mean().dropna()


# In[106]:


df_daily.describe()


# In[107]:



df_daily.info()


# Next creatre train and test datasets from this daily data, then scale and define a function to create the right inputs for regression models

# In[108]:


# kind of Baseline Model
# I changed the dependent variable from total load forecast to total load actual
df_daily.head()

df_daily['total load actual']

df_daily.shape

labels = np.array(df_daily['total load actual'])
df_joined_pred_daily = df_daily.drop('total load actual', axis = 1)
df_joined_pred_daily

train_instances, test_instances, train_labels, test_labels = train_test_split (df_joined_pred_daily, labels, test_size = 0.20, random_state = 4180)

print('Training instances shape:', train_instances.shape)
print('Training labels shape:', train_labels.shape)
print('Testing instances shape:', test_instances.shape)
print('Testing labels shape:', test_labels.shape)


# In[109]:


# Linear Regression
regr = linear_model.LinearRegression()
model = regr.fit(train_instances, train_labels)

test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)

mape(test_labels, test_predictions)


# In[110]:


# KNN Regression
regr = KNeighborsRegressor (n_neighbors = 5) #number of neighbors is arbitrary
model = regr.fit(train_instances, train_labels)
test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)

mape(test_labels, test_predictions)


# In[111]:


# Decision Tree Regression
regr = DecisionTreeRegressor(random_state = 42)
model = regr.fit(train_instances, train_labels)
test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)

mape(test_labels, test_predictions)


# In[112]:


# Random Forest Regression
regr = RandomForestRegressor(random_state = 42, n_estimators = 10)
model = regr.fit(train_instances, train_labels)
test_predictions = model.predict(test_instances)
evaluate_results(test_labels, test_predictions)

mape(test_labels, test_predictions)


# # --> Q3 Using Classification, how to accurately forecast daily energy demand

# In[113]:


df1['total generation'].hist()

df1['total generation'].sum()

df1['total load actual'].sum()

df1['total generation'].sum() - df1['total load actual'].sum()


# Negative value mean over the 4 year that there is shortage of electricity

# In[114]:


df_joined_prediction = df_joined[['total load actual','total generation','generation fossil gas','generation fossil oil','generation hydro water reservoir','price actual','generation fossil hard coal','temp_C_max','generation other renewable','generation fossil brown coal/lignite','forecast solar day ahead','wind_speed','humidity','generation hydro run-of-river and poundage','rain_1h', 'generation nuclear', 'generation biomass']]


# In[115]:


df_joined_prediction.info()


# In[116]:


df_joined_prediction.describe()


# In[117]:


df_joined_prediction.head()


# In[118]:


df_joined_prediction.shape


# In[119]:


df_daily_class = df_joined_prediction.resample("D").mean().dropna()

df_daily_class.shape


# In[120]:


df_daily_class.head()


# In[121]:


df_daily_class.info()


# In[122]:


df_daily_class['total load actual'].mean()


# Check - 
# 
# mean = 41900405/1460 = 28698.9

# In[123]:


df_daily_class['total load actual']


# In[124]:


df_daily_class.loc[df_daily_class['total load actual']>=28700, 'total load actual']


# In[125]:


df_daily_class.loc[df_daily_class['total load actual']<28700, 'total load actual']


# Therefore there is almost no imbalance in class since both class have almost similar number of records

# In[126]:


df_daily_class.describe().T


# Choosing 28700 MW as threshold to create two classes since mean daily load is 28699 MW

# In[128]:


df_daily_class['high_load_level'] = df_daily_class['total load actual']>= 28700

df_daily_class.high_load_level.value_counts()

df_daily_class.head()


# In[132]:


from pandas._libs.tslibs.vectorized import normalize_i8_timestamps

def normalize(x):
  y = (x-min(x))/(max(x)-min(x))
  return y

clmn = ['total load actual','total generation','generation fossil gas','generation fossil oil','generation hydro water reservoir','price actual','generation fossil hard coal','temp_C_max','generation other renewable','generation fossil brown coal/lignite','forecast solar day ahead','wind_speed','humidity','generation hydro run-of-river and poundage','rain_1h', 'generation nuclear', 'generation biomass']

normal_data = df_daily_class[clmn]
normal_df_daily_class = normal_data.apply(normalize)
normal_df_daily_class['high_load_level'] = df_daily_class['high_load_level']


# In[133]:


display(normal_df_daily_class.head())


# In[134]:


# Therefore, data is normalized since all values are between 0 and 1.

training_test_data = normal_df_daily_class.copy()

independent = training_test_data.loc[:,training_test_data.columns != 'high_load_level']

dependent = training_test_data['high_load_level']


# In[135]:


#Split the dataset into training and test based on quality (Ratio 80:20)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(independent, dependent,test_size=0.20,random_state=4180)

print(x_train.shape)
print(x_test.shape)


# In[136]:


import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics._plot.confusion_matrix import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# In[137]:


logistic_regression = LogisticRegression(solver = 'liblinear', random_state=4180)

logistic_regression.fit(x_train,y_train.values.ravel())

log_prediction = logistic_regression.predict(x_test)

log_prediction_df = pd.DataFrame(log_prediction)

log_prediction_df.head()


# In[138]:


#Calculating accuracy
logistic_regression_accuracy=round(metrics.accuracy_score(y_test,log_prediction)*100,2)
logistic_regression_accuracy


# In[139]:


# Therefore, accuracy of Logistic Regression model is almost 94.18%

print(y_train.shape)
print(y_test.shape)

y_train.info()

y_train.head()

y_test.head()


# In[140]:


# Confusion Matrix for Logistic Regression
actual = y_test
predicted = log_prediction
Confusion_Matrix_Log_Reg = confusion_matrix(actual, predicted)
Confusion_Matrix_Log_Reg

KNN = KNeighborsClassifier(n_neighbors=5)

KNN.fit(x_train, y_train.values.ravel())

KNN_prediction = KNN.predict(x_test)

KNN_prediction_df = pd.DataFrame(KNN_prediction)
KNN_prediction_df.head()


# In[141]:


#Calculating accuracy
KNN_accuracy=round(metrics.accuracy_score(y_test,KNN_prediction)*100,2)
KNN_accuracy


# In[142]:


# Therefore, accuracy of KNN model is about 91.1% which is worse than logistic Regression model.

# Confusion Matrix for KNN Classifier
actual = y_test
predicted = KNN_prediction
Confusion_Matrix_KNN= confusion_matrix(actual, predicted)
Confusion_Matrix_KNN


# In[143]:


#Logistic Regression Model
print("Logistic Regression Model")
LR_Accuracy = metrics.accuracy_score(y_test, log_prediction)
print("Logistic Regression Accuracy:", LR_Accuracy)
LR_Precision = metrics.precision_score(y_test, log_prediction, average='weighted', zero_division=0)
print("Logistic Regression Precision:", LR_Precision)
LR_Recall = metrics.recall_score(y_test, log_prediction, average='weighted', zero_division=0)
print("Logistic Regression Recall:", LR_Recall)


# In[144]:


#KNN Model
print("KNN Model")
KNN_Accuracy = metrics.accuracy_score(y_test, KNN_prediction)
print("KNN Accuracy:", KNN_Accuracy)
KNN_Precision = metrics.precision_score(y_test, KNN_prediction, average='weighted', zero_division=0)
print("KNN Precision:", KNN_Precision)
KNN_Recall = metrics.recall_score(y_test, KNN_prediction, average='weighted', zero_division=0)
print("KNN Recall:", KNN_Recall)


# Accuracy = (TP + TN) / (TP + FP + TN + FN) where TP = True positives, TN = True negatives, FP = False positives, and FN = False negatives. Accuracy measures the ratio of correct predictions made by the model to total predictions.
# Accuracy of Logistic Regression model is almost 93.5% whihc is higher than KNN classification model Accuracy of 90%
# 
# Precision = TP / (TP + FP) Precision is the accuracy of predicting positive class.
# Precision of Logistic Regression is almost 93.5% which is higher than KNN classifictaion Precision of almost 90%
# 
# Recall = TP / (TP + FN) Recall is the True positive rate which means it is the ratio or true postive to all postive.
# Recall of Logistic Regression is almost 93.5% which is higher than KNN classifictaion Recall of almost 90%
# 
# All three performance measures are better for Logistic Regression model. Therefore, Logistic Regression perforem better than KNN classification for this dataset

# In[145]:


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


# In[146]:


# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(x_train, y_train)


# In[147]:


# Make predictions on the testing data
DT_pred = clf.predict(x_test)


# In[148]:


# Evaluate the accuracy of the classifier
accuracy = metrics.accuracy_score(y_test, DT_pred)
print("Accuracy:", accuracy)

actual = y_test
predicted = DT_pred
Confusion_Matrix_DT= confusion_matrix(actual, predicted)
Confusion_Matrix_DT

print("Decision Tree Classification Model Accuracy")
DT_Accuracy = metrics.accuracy_score(y_test, DT_pred)
print("Decision Tree Classification Accuracy:", DT_Accuracy)
DT_Precision = metrics.precision_score(y_test, DT_pred, average='weighted', zero_division=0)
print("Decision Tree Classification Precision:", DT_Precision)
DT_Recall = metrics.recall_score(y_test, DT_pred, average='weighted', zero_division=0)
print("Decision Tree Classification Recall:", DT_Recall)


# # there seem to be issue with Decision tree classifier because all three performance measures are 100%

# In[170]:


# Importing necessary libraries for data manipulation, model building, and evaluation
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


# Let's add two new features: total_energy_city (total energy produced by each city) and energy_source (categorizing energy sources as 'Greener' or 'Legacy').

# In[171]:


# Add new feature: Total energy produced by the city
df_joined['total_energy_city'] = df_joined[['generation solar', 'generation waste', 'generation wind onshore']].sum(axis=1)

# Function to classify energy source as Greener (Renewable Energy) or Legacy (Fossil Fuel)
def classify_energy(row):
    if row['generation solar'] > 0 or row['generation wind onshore'] > 0:
        return 'Greener (Renewable Energy)'
    else:
        return 'Legacy (Fossil Fuel)'

# Apply the function to the dataframe
df_joined['energy_source'] = df_joined.apply(classify_energy, axis=1)


# In[172]:


df_joined


# Self-Reliance Based on Energy Threshold
# We will define a threshold for self-reliance. If the total energy produced by the city exceeds the threshold, we classify the city as self-reliant.

# In[174]:


# Define energy threshold (mean of total generation as a threshold for self-reliance)
energy_threshold = df_joined['total_generation'].mean()


# In[175]:



# Add a column indicating whether the city is self-reliant based on the total energy produced
df_joined['self_reliant'] = df_joined['total_energy_city'] >= energy_threshold


# In[176]:



# Display how many cities are self-reliant and how many are not
print(df_joined['self_reliant'].value_counts())


# # Time-Series Forecasting Using ARIMA

# In[178]:


# Ensure the 'total_generation' column is in float format
df_joined['total_generation'] = df_joined['total_generation'].astype(float)

# Resample the data to daily frequency and sum the total generation for each day
df_joined_daily = df_joined.resample('D').sum()['total_generation']

# Split the data into training and testing sets (80% for training, 20% for testing)
train_size = int(len(df_joined_daily) * 0.8)
train, test = df_joined_daily[:train_size], df_joined_daily[train_size:]

# Fit ARIMA model (we will use a simple order of (5,1,0) – this can be adjusted based on ACF and PACF plots)
model = sm.tsa.ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()

# Forecast the next values using the ARIMA model
predictions = model_fit.forecast(steps=len(test))

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, predictions, label='ARIMA Predictions', color='red')
plt.legend()
plt.title('ARIMA Model Forecast for Total Generation')
plt.show()


# # SARIMAX Model (Seasonal ARIMA with Exogenous Variables)
# SARIMAX extends ARIMA by including seasonal effects and exogenous variables (such as solar, wind, and price data). We will apply SARIMAX to improve the forecasting model.

# In[179]:


# Exogenous variables (using solar generation, wind generation, and price as inputs)
exog_vars = df_joined[['generation solar', 'generation wind onshore', 'price actual']].resample('D').sum()

# Fit SARIMAX model (with seasonal components and exogenous variables)
sarimax_model = sm.tsa.SARIMAX(train, exog=exog_vars[:train_size], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarimax_fit = sarimax_model.fit()

# Forecast using SARIMAX
sarimax_predictions = sarimax_fit.forecast(steps=len(test), exog=exog_vars[train_size:])

# Plot the results for SARIMAX
plt.figure(figsize=(10, 6))
plt.plot(train, label='Train Data')
plt.plot(test, label='Test Data')
plt.plot(test.index, sarimax_predictions, label='SARIMAX Predictions', color='green')
plt.legend()
plt.title('SARIMAX Model Forecast for Total Generation')
plt.show()


# In[180]:


# Function to evaluate the model performance
def evaluate_forecasting_model(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = math.sqrt(mse)
    r2 = r2_score(true_values, predicted_values)
    
    # Print the evaluation metrics
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
    print(f'R-squared (R2): {r2:.2f}')

# Evaluate ARIMA model
print("Evaluation for ARIMA Model:")
evaluate_forecasting_model(test, predictions)

# Evaluate SARIMAX model
print("Evaluation for SARIMAX Model:")
evaluate_forecasting_model(test, sarimax_predictions)


# In[182]:


# Select features for Random Forest model (energy generation and weather-related features)
X = df_joined[['generation solar', 'generation waste', 'generation wind onshore', 'humidity']]
y = df_joined['total_generation']

# Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions using Random Forest model
rf_predictions = rf_model.predict(X_test)

# Evaluate Random Forest model performance
print("Evaluation for Random Forest Model:")
evaluate_forecasting_model(y_test, rf_predictions)


# In[ ]:




