import numpy as np
import pandas as pd

# Step 1: Read measurements dataset and delete first two rows (set row #2 as the column row)
from IPython.core.display import display

df_measures = pd.read_csv('./data/January MEASURE.csv', header=2)
# Remove empty columns
df_measures.dropna(how='all', axis=1, inplace=True)

# Step 2: Read critical values of features/measurements
df_critical = pd.read_csv('./data/CRITICAL FEA.csv', index_col='FEATURE')

#print(df_measures)
#print(df_critical)

def increase_counter(counter, serial_number, is_critical):
    counter[serial_number]['total'] += 1
    if is_critical:
        counter[serial_number]['critical'] += 1


def count_feature_values():
    global df_measures, df_critical

    count_good = {}
    count_bad = {}
    count_ignored = {}
    for i in df_measures.index:  # Go through all serial numbers one by one
        curr_part = df_measures.loc[i]
        sn = curr_part['SN']
        # initialize the counters
        if sn not in count_good: count_good[sn] = {'total': 0, 'critical': 0}
        if sn not in count_bad: count_bad[sn] = {'total': 0, 'critical': 0}
        if sn not in count_ignored: count_ignored[sn] = {'total': 0, 'critical': 0}

        # Go through all the features for the current part one by one (column 4 is the first feature)
        for feature in df_measures.columns[3:]:
            curr_val = curr_part[feature]
            is_critical = df_critical.loc[feature]['CRITICAL'] == 1

            if np.isnan(curr_val):  # If value is null simply ignore it
                increase_counter(count_ignored, sn, is_critical)
                break

            lsl = df_critical.loc[feature]['LSL']
            usl = df_critical.loc[feature]['USL']

            if lsl <= curr_val <= usl:
                increase_counter(count_good, sn, is_critical)
            else:
                increase_counter(count_bad, sn, is_critical)

    return count_good, count_bad, count_ignored


# Calculate and store the different quantities of good, bad and ignores features for each part
counter_good, counter_bad, counter_ignored = count_feature_values()

### TASK 1.02 ###
### Calculate the critical feature and overall pass rate (good count/total count) for each part by SN ###

def create_pass_rate_table(count_good, count_bad, count_ignored):
    global df_measures, df_critical

    df_rates = pd.DataFrame()
    for sn in count_good.keys():
        total_good = count_good[sn]['total']
        total_bad = count_bad[sn]['total']
        total_ignored = count_ignored[sn]['total']
        bad_critical = count_bad[sn]['critical']
        good_critical = count_good[sn]['critical']
        pass_rate = total_good / (total_good + total_bad)
        critical_pass_rate = good_critical / (good_critical + bad_critical)

        row = pd.Series(
            data={'Good': total_good, 'Bad': total_bad, 'Bad Critical': bad_critical, 'Ignored': total_ignored,
                  'Pass Rate': pass_rate, 'Critical Pass Rate': critical_pass_rate}, name=sn)
        df_rates = df_rates.append(row)

    # Change columns data types to correct types (counters as integers)
    df_rates = df_rates.astype({'Good': int, 'Bad': int, 'Bad Critical': int, 'Ignored': int, 'Pass Rate': float, 'Critical Pass Rate': float})
    return df_rates


df_rates = create_pass_rate_table(counter_good, counter_bad, counter_ignored)
df_rates = df_rates.reset_index()
df_rates = df_rates.rename(columns ={'index':'SN'})
#print(df_rates.head())
df_rates.to_csv('./process/df_passrate.csv')