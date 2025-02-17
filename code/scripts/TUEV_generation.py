import os
import mne
import numpy as np
import pickle
import random
from tqdm import tqdm

def BuildEvents_100s_128Hz(signals, times, EventData):
    """
    将原 16通道信号 + 事件信息, 提取固定 100秒窗口(中心对齐).

    - signals: (16, n_times)
    - times:   (n_times,) 原始时间戳(已重采样到128Hz)
    - EventData: (numEvents, 4), [channel, start_time, end_time, label]

    返回:
        features: shape = (numEvents, 16, 128*100 = 12800)
        offending_channel: (numEvents, 1)
        labels: (numEvents, 1)
    """
    [numEvents, _] = EventData.shape
    fs = 128.0    # 重采样后固定128Hz
    #seg_len = int(fs * 100)  # 100秒 => 12800
    seg_len = int(fs * 5)
    half_len = seg_len // 2  # 一半是6400

    [numChan, _] = signals.shape

    # 为防止索引越界, 拼接3倍signals (与原TUEV类似)
    offset = signals.shape[1]
    signals = np.concatenate([signals, signals, signals], axis=1)

    # 分配空间
    features = np.zeros([numEvents, numChan, seg_len])
    offending_channel = np.zeros([numEvents, 1])
    labels = np.zeros([numEvents, 1])

    for i in range(numEvents):
        chan = int(EventData[i, 0])  # channel index
        start_time = EventData[i, 1]
        end_time   = EventData[i, 2]
        lbl        = int(EventData[i, 3])

        # 事件中心(秒)
        center_time = (start_time + end_time) / 2.0

        # 找到 times 中 >= center_time 的第一个索引
        center_idx = np.where(times >= center_time)[0][0]

        # 在三倍signals上找到对应起止
        seg_start = offset + center_idx - half_len
        seg_end = seg_start + seg_len  # seg_start+12800

        # 截取100秒
        features[i, :] = signals[:, seg_start:seg_end]
        offending_channel[i, 0] = chan
        labels[i, 0] = lbl

    return features, offending_channel, labels


def convert_signals(signals, Rawdata):
    """
    与原TUEV一样的16通道双极差分.
    """
    signal_names = {
        k: v
        for (k, v) in zip(Rawdata.info["ch_names"], range(len(Rawdata.info["ch_names"])))
    }
    new_signals = np.vstack((
        signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F7-REF"]],  # 0
        signals[signal_names["EEG F7-REF"]]  - signals[signal_names["EEG T3-REF"]],  # 1
        signals[signal_names["EEG T3-REF"]]  - signals[signal_names["EEG T5-REF"]],  # 2
        signals[signal_names["EEG T5-REF"]]  - signals[signal_names["EEG O1-REF"]],  # 3
        signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F8-REF"]],  # 4
        signals[signal_names["EEG F8-REF"]]  - signals[signal_names["EEG T4-REF"]],  # 5
        signals[signal_names["EEG T4-REF"]]  - signals[signal_names["EEG T6-REF"]],  # 6
        signals[signal_names["EEG T6-REF"]]  - signals[signal_names["EEG O2-REF"]],  # 7
        signals[signal_names["EEG FP1-REF"]] - signals[signal_names["EEG F3-REF"]],  # 8
        signals[signal_names["EEG F3-REF"]]  - signals[signal_names["EEG C3-REF"]],  # 9
        signals[signal_names["EEG C3-REF"]]  - signals[signal_names["EEG P3-REF"]],  # 10
        signals[signal_names["EEG P3-REF"]]  - signals[signal_names["EEG O1-REF"]],  # 11
        signals[signal_names["EEG FP2-REF"]] - signals[signal_names["EEG F4-REF"]],  # 12
        signals[signal_names["EEG F4-REF"]]  - signals[signal_names["EEG C4-REF"]],  # 13
        signals[signal_names["EEG C4-REF"]]  - signals[signal_names["EEG P4-REF"]],  # 14
        signals[signal_names["EEG P4-REF"]]  - signals[signal_names["EEG O2-REF"]],  # 15
    ))
    return new_signals


def readEDF(fileName):
    """
    读取 .edf 并重采样到128Hz, 同时解析 .rec 文件事件信息.
    """
    # 使用 mne 读取, preload=True 方便后续处理
    Rawdata = mne.io.read_raw_edf(fileName, preload=True)
    # 关键改动: 重采样到128Hz
    Rawdata.resample(128)

    signals, times = Rawdata[:]

    # 读取 .rec 文件 (假设与 .edf 同名同目录)
    RecFile = fileName[:-4] + ".rec"
    eventData = np.genfromtxt(RecFile, delimiter=",")

    return signals, times, eventData, Rawdata


def save_pickle(obj, filename):
    """
    保存到pkl文件
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def process_edf_list(edf_list, out_dir):
    """
    给定一批 .edf 文件列表, 对每个文件执行:
      1) readEDF -> 得到(重采样后) signals, times, eventData
      2) convert_signals -> 16通道差分
      3) BuildEvents_100s_128Hz -> 得到 100秒片段
      4) 保存每个事件段为 pkl
    """
    os.makedirs(out_dir, exist_ok=True)

    for edf_path in tqdm(edf_list, desc=f"Processing {out_dir}"):
        fname = os.path.basename(edf_path)
        fname_noext = os.path.splitext(fname)[0]

        try:
            signals, times, eventData, Rawdata = readEDF(edf_path)
            signals = convert_signals(signals, Rawdata)
        except (ValueError, KeyError, IndexError) as e:
            print(f"[跳过] 读取失败 {edf_path}, 错误: {e}")
            continue

        signals_100s, offending_channels, labels = BuildEvents_100s_128Hz(signals, times, eventData)

        for idx, (sig_1, off_ch, lbl) in enumerate(zip(signals_100s, offending_channels, labels)):
            sample = {
                "signal": sig_1,            # shape: (16, 12800)
                "offending_channel": off_ch,# shape: (1,)
                "label": lbl,               # shape: (1,)
            }
            out_pkl = os.path.join(
                out_dir, f"{fname_noext}-{idx}.pkl"
            )
            save_pickle(sample, out_pkl)


def gather_all_edf_files(base_dir):
    """
    遍历 base_dir 下的所有子目录, 收集所有后缀 .edf 文件路径
    返回一个 list, 每个元素是 .edf 文件的完整路径
    """
    edf_files = []
    for root, dirs, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(".edf"):
                edf_files.append(os.path.join(root, fname))
    return edf_files


##########################################
# 主程序示例
##########################################

if __name__ == "__main__":
    # 1) 假设原始 TUEV 数据放在:
    root = "/data/datasets_public/TUEV/edf"
    # 目录结构:
    #  /data/datasets_public/TUEV/edf/train/....
    #  /data/datasets_public/TUEV/edf/eval/.....

    # 2) 我们要把 train 文件夹内的 .edf 文件随机打乱, 按 80/20 拆分为 train/val
    base_train_dir = os.path.join(root, "train")
    all_train_edfs = gather_all_edf_files(base_train_dir)
    print("在 train 文件夹下找到 EDF 文件数:", len(all_train_edfs))

    random.shuffle(all_train_edfs)
    split_idx = int(len(all_train_edfs) * 0.8)
    train_edfs = all_train_edfs[:split_idx]  # 80%
    val_edfs   = all_train_edfs[split_idx:]  # 20%

    # 3) eval 文件夹全部作为 "test"
    base_eval_dir = os.path.join(root, "eval")
    test_edfs = gather_all_edf_files(base_eval_dir)
    print("在 eval 文件夹下找到 EDF 文件数:", len(test_edfs))

    # 4) 定义输出目录
    #    - processed_train_128hz_100s
    #    - processed_val_128hz_100s
    #    - processed_test_128hz_100s
    out_root = os.path.join(root, "processed_128hz_640seqlen_JH")
    train_out_dir = os.path.join(out_root, "train")
    val_out_dir   = os.path.join(out_root, "val")
    test_out_dir  = os.path.join(out_root, "test")

    # 5) 分别处理 train/ val/ test
    print(">>> 处理 TRAIN (80%)")
    process_edf_list(train_edfs, train_out_dir)

    print(">>> 处理 VAL   (20%)")
    process_edf_list(val_edfs, val_out_dir)

    print(">>> 处理 TEST (eval 文件夹)")
    process_edf_list(test_edfs, test_out_dir)

    print(">>> 全部处理完毕!")