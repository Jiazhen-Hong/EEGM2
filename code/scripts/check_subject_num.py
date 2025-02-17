import numpy as np

data_name = "Alpha" #Alpha, Attention, Crowdsource, STEW, DriverDistraction, DREAMER,
file_path = f'/data/data_downstream_task/{data_name}/{data_name}_id.npy'

# 读取数据
data_id = np.load(file_path, allow_pickle=True)

# 计算唯一的 subject 数量
unique_subjects = np.unique(data_id)
num_subjects = len(unique_subjects)

print(f"Total number of unique subjects: {num_subjects}")