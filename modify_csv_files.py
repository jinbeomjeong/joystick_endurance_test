import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor


def process_file(file_path):
    data = pd.read_csv(file_path, encoding='cp949').iloc[9:]
    data.columns = data_column_names
    data.reset_index(drop=True, inplace=True)
    data.to_csv('D:\\Workspace\\Python\\joystick_endurance_test\\data' + os.sep + os.path.basename(file_path))


def task(path_list):
    with ThreadPoolExecutor() as executor:
        executor.map(process_file, path_list)


if __name__ == '__main__':
    data_root_path = "d:\\joystick"
    sample_names = ['RAW DATA 1']

    file_path_list = list()

    for sample_name in sample_names:
        for file in os.listdir(os.path.join(data_root_path, sample_name, 'Durability')):
            if file.endswith('.CSV'):
                file_path_list.append(os.path.join(data_root_path, sample_name, 'Durability', file))

    file_path_list.sort()

    data_column_names = ['port_1_p(bar)', 'port_2_p(bar)', 'port_3_p(bar)', 'port_4_p(bar)', 'support_p(bar)',
                         'tank_p(bar)', 'temp(c)']

    task(file_path_list)
