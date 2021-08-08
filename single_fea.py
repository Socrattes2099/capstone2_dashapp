## 4.02
# Predict a single feature status

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



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
filter_tol.to_csv('./process/filter_tol.csv')

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
df_measure_cleaned2.reset_index(drop=True, inplace=True)
#print('df_measure_cleaned2: \n', df_measure_cleaned2)
df_measure_cleaned2.to_csv('./process/df_measure_cleaned2.csv')

def single_fea(feature_name):
        from statistics import stdev

        # Ask user to select one feature
        #user_input_1 = input('Please enter a feature name: ')

        #Feature_name = user_input_1
        Feature_name = feature_name
        #Feature_name = 'B008_TP_ABC[Y]'        # for test use only

        OneFea = df_measure_cleaned2[Feature_name]
        OneTol = filter_tol[Feature_name]

        #print(OneFea)
        #print(OneTol)

        # Predict based on linear regression
        X = list(range(0,len(OneFea)))
        Y = OneFea
        y_fea = Y.values.reshape(-1,1)
        x_fea = np.asarray(X).reshape(-1,1)
        regressor = LinearRegression()
        regressor.fit(x_fea, y_fea)
        X_pred = np.array(list(range(0,2000))).reshape(-1, 1)
        y_pred = regressor.predict(X_pred)
        # Standard deviation of last 100 parts
        stdev = np.std(y_fea[-100:,:])
        # Prediction of 3 sigma range
        y_pred_low = y_pred-3*stdev
        y_pred_hi = y_pred+3*stdev

        # Create df for current X and Y
        df_LR_single_now = pd.DataFrame(columns=['X', 'Y'])
        df_LR_single_now['X'] = X
        df_LR_single_now['Y'] = Y

        # Create df for prediction
        df_LR_single_pred = pd.DataFrame(X_pred, columns=['X_pred'])
        df_LR_single_pred['y_pred'] = y_pred
        df_LR_single_pred['y_pred_low'] = y_pred_low
        df_LR_single_pred['y_pred_hi'] = y_pred_hi

        print(df_LR_single_now)
        print(df_LR_single_pred)
        df_LR_single_now.to_csv('./process/df_LR_single_now.csv')
        df_LR_single_pred.to_csv('./process/df_LR_single_pred.csv')
        OneTol.to_csv('./process/OneTol.csv')

        '''
        # Plotting the graph
        plt.figure(figsize=(20,10))
        plt.plot(x_fea, y_fea, 'mediumblue')
        plt.plot(X_pred, y_pred,'r-.')
        plt.plot(X_pred, y_pred_low,'m-.')
        plt.plot(X_pred, y_pred_hi,'m-.')
        plt.xlabel('Sequential Number', fontdict={'fontweight':'bold','fontsize':20})
        plt.ylabel('Deviation', fontdict={'fontweight':'bold','fontsize':20})
        plt.axhline(y=float(OneTol[1]), color='orange', linestyle='--')
        plt.axhline(y=float(OneTol[2]), color='orange', linestyle='--')
        plt.legend(['Current', 'Forecast', 'Forecast-3*sigma', 'Forecast+3*sigma', 'LSL', 'USL'])
        plt.title('Prediction based on LR: '+ Feature_name, fontdict={'fontweight':'bold','fontsize':30})
        plt.savefig(('Feature Moving Range Prediction'), dpi=300)
        plt.show()
        '''