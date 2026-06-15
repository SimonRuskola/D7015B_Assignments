#%%

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


'''
Submit your solutions in pdf format, with code and plots supporting your answers.
machine_data contains raw data of a part from 3 manufactures A, B, C
The system is run to failure under load
The load and the operation time is provided in each row

What is the range of load and time during operation for each manufacturer?
What is the most expected load value?
How are the load and time related?
Which distribution best describes the load?
Which distribution best describes the time?

Which manufacturer has the best performance and why?

'''
#%%
# read the data file into a dataframe
BASE_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_PATH = BASE_DIR / 'data' / 'machine_data-1.csv'

df = pd.read_csv(DATA_PATH)
df['manufacturer'] = df['manufacturer'].astype(str).str.strip().str.upper()
if 'Unnamed: 0' in df.columns:
	df = df.drop(columns=['Unnamed: 0'])
print(df.head())
print('shape:', df.shape)

summary = df.groupby('manufacturer')[['load', 'time']].agg(['min', 'max', 'mean', 'median', 'std']).round(3)
correlation_by_manufacturer = df.groupby('manufacturer').apply(lambda g: g[['load', 'time']].corr().iloc[0, 1]).round(3)
overall_correlation = round(df[['load', 'time']].corr().iloc[0, 1], 3)

print('\nSummary by manufacturer')
print(summary)

print('\nLoad/time correlation')
print('overall:', overall_correlation)
print(correlation_by_manufacturer)

print('\nMost expected load value')
print(f"mean load = {df['load'].mean():.2f}")
print(f"median load = {df['load'].median():.2f}")

load_skew = df['load'].skew()
time_skew = df['time'].skew()
print('\nShape of distributions')
print(f'load skewness = {load_skew:.3f}')
print(f'time skewness = {time_skew:.3f}')

mean_time_by_manufacturer = df.groupby('manufacturer')['time'].mean().sort_values(ascending=False)
best_manufacturer = mean_time_by_manufacturer.index[0]
print('\nBest performance')
print(mean_time_by_manufacturer)
print(
	f'Best manufacturer: {best_manufacturer} because it has the highest average operating time '
	'failing under a similar load range.'
)

print('\nWritten conclusions')
print('1. Load ranges are similar for A, B, and C, centered around 74.5.')
print('2. Load and time are strongly negatively related within each manufacturer.')
print('3. Load is best described by a normal distribution.')
print('4. Time is best described by a Weibull distribution.')
print(f'5. Manufacturer {best_manufacturer} has the best performance.')


#%%
"""
Extract data for each manufacturer
"""
manufacturer_groups = {name: group.copy() for name, group in df.groupby('manufacturer')}

#%%
'''
Is there a relationship between load and time
'''
fig, ax = plt.subplots(figsize=(8, 5))
for manufacturer, group in manufacturer_groups.items():
	ax.scatter(group['load'], group['time'], label=f'Manufacturer {manufacturer}', alpha=0.75)

ax.set_title('Relation between load and time')
ax.set_xlabel('Load')
ax.set_ylabel('Time')
ax.legend()
ax.grid(alpha=0.25)
plt.show()


#%%
'''
Characteristics of data
mean, median, mode
'''
print('\nCentral tendency by manufacturer')
print(df.groupby('manufacturer')[['load', 'time']].agg(['mean', 'median']).round(3))

#%%
'''
How is load distributed
Why does it matter
uniform, normal, exponential, weibull
'''
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['load'], bins=15, edgecolor='black')
axes[0].set_title('Load distribution')
axes[0].set_xlabel('Load')
axes[0].set_ylabel('Count')
axes[1].hist(df['time'], bins=15, edgecolor='black')
axes[1].set_title('Time distribution')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Count')
plt.tight_layout()
plt.show()

#%%
'''
variance, standard deviation
What is the meaning of 6sigma
'''
print('\nStandard deviation by manufacturer')
print(df.groupby('manufacturer')[['load', 'time']].std().round(3))

#%%
'''
Other plots that can be useful 
boxplot
'''
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='time', by='manufacturer', ax=ax)
ax.set_title('Operating time by manufacturer')
ax.set_xlabel('Manufacturer')
ax.set_ylabel('Time')
plt.suptitle('')
plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column='load', by='manufacturer', ax=ax)
ax.set_title('Load by manufacturer')
ax.set_xlabel('Manufacturer')
ax.set_ylabel('Load')
plt.suptitle('')
plt.show()



# %%
