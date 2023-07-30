import os
import pandas as pd

def combine_csv_files(directory):
    df_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath)
            df_list.append(df)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


def combine_csv_files_and_average(directory, column_name):
    df_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, skiprows=range(5, None, 6))  # only read every 6th row
            df_mean = df.groupby(df.index // 5)[column_name].mean()  # group by index and find mean
            df_list.append(df_mean)
    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df


#function to average error and std dev over each iteration


#function to average error and std dev over each radius

