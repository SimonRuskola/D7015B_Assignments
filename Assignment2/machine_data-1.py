#%%

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
df = pd.read_csv('Assignment2/data/machine_data-1.csv')
df['manufacturer'] = df['manufacturer'].astype(str).str.strip().str.upper()
if 'Unnamed: 0' in df.columns:
	df = df.drop(columns=['Unnamed: 0'])
print(df)

print(df.shape)

#%% 
"""
Drop the index
"""

#%%
"""
Extract data for a given manufacturer
"""
manufacturer = 'A'
dfa = df.loc[df['manufacturer'].eq(manufacturer)].copy()

#%%

loada = dfa['load']
timea = dfa['time']

#%%
'''
Is there a relationship between load and time
'''
plt.scatter(loada, timea)
plt.title("Relation between load and time")
plt.xlabel("Load")
plt.ylabel("Time")
plt.show()


#%%
'''
Characteristics of data
mean, median, mode
'''
dfa['load'].mean()
#%%
'''
How is load distributed
Why does it matter
uniform, normal, exponential, weibull
'''
dfa[['load']].plot(kind='hist', bins=10)

#%%
'''
variance, standard deviation
What is the meaning of 6sigma
'''
#%%
'''
Other plots that can be useful 
boxplot
'''


