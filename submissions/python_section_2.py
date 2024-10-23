import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#QUESTION NUMBER 9

def calculate_distance_matrix(file_path):
    
    df = pd.read_csv(file_path)
    
    df_pivot = df.pivot(index='id_start', columns='id_end', values='distance')
    distance_matrix = df_pivot.add(df_pivot.transpose(), fill_value=0)
    np.fill_diagonal(distance_matrix.values, 0)
    for i in range( 0 ,len(distance_matrix)):
        for j in range( 0, len(distance_matrix)):
            if np.isnan(distance_matrix.iloc[i, j]):
                distance_matrix.iloc[i, j] = distance_matrix.iloc[i, :j].add(distance_matrix.iloc[:j, j]).min()
                distance_matrix.iloc[j, i] = distance_matrix.iloc[i, j]

    return distance_matrix

distance_matrix = calculate_distance_matrix('dataset-2.csv')
print(distance_matrix.head())

#QUESTION NUMBER 10

def unroll_distance_matrix(distance_matrix):
   
    long_df = distance_matrix.reset_index().melt(id_vars='index', var_name='id_start', value_name='distance')
   
    long_df.rename(columns={'index': 'id_end'}, inplace=True)
    
   
    long_df = long_df[long_df['id_start'] != long_df['id_end']]
    
    
    long_df = long_df[long_df['distance'] > 0]

    
    long_df = long_df[['id_start','id_end',  'distance']]

    return long_df

unrolled_df = unroll_distance_matrix(da)
print(unrolled_df)

#QUESTION NUMBER 11

def find_ids_within_ten_percentage_threshold(df, reference_id):
    
    reference_avg_distance = df[df['id_start'] == reference_id]['distance'].mean()

    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    ids_within_threshold = df.groupby('id_start').filter(lambda x: lower_threshold <= x['distance'].mean() <= upper_threshold)

    return sorted(ids_within_threshold['id_start'].unique())
reference_id = 1001416 
result = find_ids_within_ten_percentage_threshold(unrolled_df, reference_id)
print(result)

#QUESTION NUMBER 12

def calculate_toll_rate(df):
    
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df

toll_rate_df = calculate_toll_rate(unrolled_df)
print(toll_rate_df)

#QUESTION NUMBER 13

from datetime import time, timedelta

def calculate_time_based_toll_rates(df):
    
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    discounts = {
        'weekdays': {
            'morning': (time(0, 0), time(10, 0), 0.8),
            'day': (time(10, 0), time(18, 0), 1.2),
            'evening': (time(18, 0), time(23, 59), 0.8),
        },
        'weekends': (time(0, 0), time(23, 59), 0.7)
    }

    rows = []

    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distances = row[['moto', 'car', 'rv', 'bus', 'truck']].values

        for day in days_of_week:
            for interval in discounts['weekdays'].values():
                start_time, end_time, factor = interval
                adjusted_rates = distances * factor
                rows.append([day, start_time, day, end_time] + list(adjusted_rates))

            for day in ['Saturday', 'Sunday']:
                start_time, end_time, factor = discounts['weekends']
                adjusted_rates = distances * factor
                rows.append([day, start_time, day, end_time] + list(adjusted_rates))

    columns = ['start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck']
    result_df = pd.DataFrame(rows, columns=columns)

    return result_df


time_based_toll_df = calculate_time_based_toll_rates(toll_rate_df)
print(time_based_toll_df)