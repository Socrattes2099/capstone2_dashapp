### 4.04.2
# Use Linear Regression to Prredict future part status

import pandas as pd
import numpy as np

# Import measurement with tolerance
df_measure_tol = pd.read_csv('./data/January MEASURE.csv', header=None)
# Move title row to the top
df_measure_tol = pd.concat([df_measure_tol.iloc[[2],:], df_measure_tol.drop(2, axis=0)], axis=0)
# re-index
df_measure_tol = df_measure_tol.reset_index(drop=True)
#grab the first row for the header
new_header = df_measure_tol.iloc[0]
#take the data less the header row
df_measure_tol = df_measure_tol[1:]
df_measure_tol.columns = new_header
#drop columns that has no LSL and USL data
df_measure_tol.drop(df_measure_tol.columns[1:3], axis=1, inplace=True)
#Drop empty columns
df_measure_tol.dropna(axis=1, inplace=True)
# Drop SN column, saving only values
df_measure_tol = df_measure_tol.drop(['SN'], axis=1)
#print(df_measure_tol.shape)
#print('df_measure_tol: \n', df_measure_tol)
#df_measure_tol.to_csv('df_measure_tol.csv')

# Save tol to filter_tol
filter_df = df_measure_tol.iloc[2:, 1:].astype(float)
filter_tol = df_measure_tol.iloc[0:2, :]
filter_tol = pd.DataFrame(filter_tol)
#print(filter_tol.shape)
#print('filter_tol: \n', filter_tol)
#filter_tol.to_csv('filter_tol.csv')

# Clean outliers in df_measure_tol values
def replace_outlier(val, mean, std):
    if val > mean + 3*std:
        return mean + 3*std
    elif val < mean - 3*std:
        return mean - 3*std
    return val

filter_df = df_measure_tol.iloc[2:, :].astype(float)
#print('filter_df: \n', filter_df)

for col in filter_df.columns:
    mean = filter_df[col].mean()
    std_dev = filter_df[col].std(axis=0)
    filter_df[col] = filter_df[col].map(lambda x: replace_outlier(x, mean, std_dev))

df_measure_cleaned2 = filter_df
#print('df_measure_cleaned2: \n', df_measure_cleaned2)
#df_measure_cleaned2.to_csv('df_measure_cleaned2.csv')

# Use Linear Regression to predict the future 2000th, 3000th part status
from sklearn.linear_model import LinearRegression

df_LR = pd.DataFrame()
# For loop to predict each feature column by column
for column in df_measure_cleaned2:
    X = list(range(0,len(df_measure_cleaned2)))
    Y = df_measure_cleaned2[column]
    y1 = Y.values.reshape(-1,1)
    x1 = np.asarray(X).reshape(-1,1)
    regressor = LinearRegression()
    regressor.fit(x1, y1)
    y_new = []
    y_i = []
    for i in range(1000,3000,1000):
        y_new.append(regressor.predict([[i]]))
        y_i.append(i)
    df_LR[column] = y_new

#print('df_LR: \n', df_LR)
#df_LR.to_csv('df_LR.csv')

# Calculate last X number of parts average
last_n = 0
df_mean = df_measure_cleaned2.iloc[last_n:,:].mean()
#print('df_mean: \n', df_mean)
#df_mean.to_csv('df_mean.csv')

# Prepare LR prediction values
df_LR_T = df_LR.T
df_LR_T.columns=['2k_th', '3k_th']
df_LR_T.reset_index(level=0, inplace=True)
df_LR_T = df_LR_T.rename(columns ={'index':'Fea'})

# Prepare current mean
df_mean = pd.DataFrame(df_mean)
df_mean.columns=['Mean']
df_mean.reset_index(level=0, inplace=True)
df_mean = df_mean.rename(columns ={0:'Fea'})

# Prepare tol
tol = df_measure_tol.iloc[0:2, 0:].astype(float)
tol_T = tol.T
tol_T.reset_index(level=0, inplace=True)
tol_T = tol_T.rename(columns ={0:'Fea', 1:'LSL', 2:'USL'})

# Combine mean with LR prediction values and tol
df_combined_1 = pd.merge(df_LR_T, df_mean, on='Fea')
df_combined_2 = pd.merge(df_combined_1, tol_T, on='Fea')

# Change to float
df_combined_2[['2k_th','3k_th']] = df_combined_2[['2k_th','3k_th']].astype(float)

#print(tol_T.shape)
#print(df_combined_1.shape)
#print(df_combined_2.shape)

df_combined_2.to_csv('./process/df_combined_2.csv')



