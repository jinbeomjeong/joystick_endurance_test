import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support
from tqdm import tqdm


data_root_path = "d:\\joystick"
sample_name = 'RAW DATA 3'

file_path_list = list()

for file in os.listdir(os.path.join(data_root_path, sample_name, 'Durability')):
    if file.endswith('.CSV'):
        file_path_list.append(os.path.join(data_root_path, sample_name, 'Durability', file))

file_path_list.sort()

data_column_names = ['port_1_p(bar)', 'port_2_p(bar)', 'port_3_p(bar)', 'port_4_p(bar)', 'support_p(bar)', 'tank_p(bar)', 'temp(c)']
float32_target_var = ['port_1_p(bar)', 'port_2_p(bar)', 'port_3_p(bar)', 'port_4_p(bar)','support_p(bar)', 'tank_p(bar)', 'time(hour)']
uint8_target_var = ['temp(c)']


def load_csv(path_list):
    data = pd.read_csv(path_list, encoding='cp949').iloc[9:]
    data.columns = data_column_names

    return data


def load_csv_executor(path_list):
    with ProcessPoolExecutor(max_workers=int(os.cpu_count())) as executor:
        df_list = list(tqdm(executor.map(load_csv, path_list), desc='dataframe loading...', total=len(path_list)))

    return pd.concat(df_list)


if __name__ == '__main__':
    freeze_support()

    result_df = load_csv_executor(file_path_list)
    result_df['time(hour)'] = np.arange(result_df.shape[0]) / (100 * 3600)
    result_df.reset_index(inplace=True, drop=True)

    for var in tqdm(float32_target_var, desc='float32 converting...'):
        result_df[var] = result_df[var].astype(np.float32)

    for var in tqdm(uint8_target_var, desc='uint8 converting...'):
        result_df[var] = result_df[var].astype(np.float32).astype(np.uint8)

    result_df.to_parquet('raw_data_3.parquet')

    print(result_df.info())
