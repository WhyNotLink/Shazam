import os
from scipy.io import wavfile
import numpy as np
import pydub
import librosa
import json
import hashlib
import matplotlib.pyplot as plt

"""
    generate data and rate from audio file
    make sure data is mono and float64
    make sure data peak normalized

"""
def func1(audio_path,):
    filename = os.path.basename(audio_path)
    song_name = os.path.splitext(filename)[0]
    print(f"Processing file: {filename}, song name: {song_name}")

    rate, data = wavfile.read(audio_path)
    if data.ndim > 1:
        left = data[:, 0].astype(np.float64)
        right = data[:, 1].astype(np.float64)
        data = (left+right) / 2.0
    else:
        data = data.astype(np.float64)
    
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    max_peek = np.max(np.abs(data))
    data = data / max_peek
    data = np.clip(data, -1.0, 1.0)

    return rate, data , song_name


"""
    use sftf to generate spectrogram
    overlap rate = 87.5% using high rate for better stft 
    get magnitude  data_freq_bins_hz  data_time_bins_s

"""
def func2(rate, data):
    sftf_dot_size = 4096
    sftf_overlap = 512
    sftf_window = np.hanning(sftf_dot_size)

    data_complex = librosa.stft(data, n_fft=sftf_dot_size, hop_length=sftf_overlap, window=sftf_window)

    data_magnitude = np.abs(data_complex)

    data_freq_bins_hz = librosa.fft_frequencies(sr=rate, n_fft=sftf_dot_size)
    data_time_bins_s = librosa.frames_to_time(np.arange(data_complex.shape[1]), sr=rate, hop_length=sftf_overlap, n_fft=sftf_dot_size)

    return data_magnitude, data_freq_bins_hz, data_time_bins_s

"""
    filter per bands and get peaks
    get data_peaks_magnitude  data_peaks_hz  data_peak_times

"""
def func3(data_magnitude, data_freq_bins_hz, data_time_bins_s):

    freq_bands = [
    (0, 125),     # 超低频
    (125, 500),   # 低频
    (500, 1000),  # 低中频
    (1000, 2000), # 中频
    (2000, 4000), # 中高频
    (4000, 8000), # 高频
    (8000, 14000),# 超高频
    (14000, 22050)# 极高频
]
    data_peaks_magnitude = []
    data_peaks_hz = []
    data_peak_times = []

    idx1 = 0;
    idx2 = 0;
    for band in freq_bands:
        data_peaks_candidates = []
        freq_min,freq_max = band
        idx2 = idx1
        for magnitude in data_magnitude:
            freq = data_freq_bins_hz[idx1]
            if freq >= freq_min and freq < freq_max:
                data_peaks_candidates.append(magnitude)
                idx1 += 1
            else:
                break
        data_peaks_magnitude.append(np.max(data_peaks_candidates,axis = 0) if data_peaks_candidates else 0)
        data_peaks_hz.append(data_freq_bins_hz[idx2+np.argmax(data_peaks_candidates,axis = 0)] if data_peaks_candidates else 0)
        data_peak_times.append(data_time_bins_s)
    return data_peaks_magnitude, data_peaks_hz, data_peak_times

"""
    switch magnitude to db
    calculate filter db by average db per time
    filter db  hz times
    make data from two to one dimension

"""
def func4(data_peaks_magnitude, data_peaks_hz, data_peak_times):

    data_peaks_db = librosa.amplitude_to_db(np.array(data_peaks_magnitude), ref=np.max)
    data_peaks_hz = np.array(data_peaks_hz)
    data_peak_times = np.array(data_peak_times)

    data_average_db = np.mean(data_peaks_db, axis=0)
    filter_data_db = []
    keep_mask = data_peaks_db >= data_average_db[np.newaxis, :]
    filter_data_db = data_peaks_db[keep_mask]
    filter_data_hz = data_peaks_hz[keep_mask]
    filter_data_times = data_peak_times[keep_mask]
    return filter_data_db, filter_data_hz, filter_data_times

"""
    generate anchors from filter data
    slect peaks as anchor points
    each anchor contain 5 peaks 

"""
def func5(filter_data_db, filter_data_hz, filter_data_times):
    sort_idx = np.argsort(filter_data_times)
    filter_data_db = filter_data_db[sort_idx]
    filter_data_hz = filter_data_hz[sort_idx]
    filter_data_times = filter_data_times[sort_idx]


    min_db = -30
    window_size = 100
    min_time_diff = 0.1
    max_pair_num = 5
    anchor_pair = []

    for i in range(len(filter_data_db)):
        if filter_data_db[i] < min_db:
            continue
        anchor_time, anchor_hz = filter_data_times[i], filter_data_hz[i]
        range_idx = min(i+window_size, len(filter_data_db))
        k = 0
        for j in range(i+1, range_idx):
            target_time, target_hz = filter_data_times[j], filter_data_hz[j]
            time_diff = target_time - anchor_time
            if time_diff < min_time_diff:
                continue
            k += 1
            if k >= max_pair_num:
                break
            anchor_pair.append((anchor_time, anchor_hz, target_hz, time_diff))   

    print(f"Generated {len(anchor_pair)} anchors.")
    return anchor_pair, filter_data_times, filter_data_hz, filter_data_db

def plot1(anchor_pair,filter_data_times, filter_data_hz, filter_data_db):
    min_db = -30

        # ===================== 提取锚点的时间/频率/分贝（匹配你的轴要求） =====================
    anchor_times = []   # 锚点X轴：时间
    anchor_hzs = []     # 锚点Y轴：频率
    anchor_dbs = []     # 锚点颜色：分贝
    for pair in anchor_pair:
        at = pair[0]  # 锚点时间
        # 精准匹配原始数据中的对应点（解决浮点精度）
        closest_idx = np.argmin(np.abs(filter_data_times - at))
        anchor_times.append(filter_data_times[closest_idx])
        anchor_hzs.append(filter_data_hz[closest_idx])
        anchor_dbs.append(filter_data_db[closest_idx])

    # ===================== 绘制最终要求的两张图（X=时间，Y=频率，颜色=分贝） =====================
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False

    # ---------- 图1：所有原始点（X=时间，Y=频率，颜色=分贝） ----------
    plt.figure(figsize=(12, 6))
    # 核心：c=filter_data_db 颜色映射分贝，cmap选冷暖色（红色=高分贝，蓝色=低分贝）
    scatter1 = plt.scatter(filter_data_times, filter_data_hz, 
                           c=filter_data_db, cmap='RdYlBu_r',  # 红-黄-蓝，反向（高分贝红，低分贝蓝）
                           s=10, alpha=0.7, vmin=-80, vmax=0)  # 分贝范围限定（-80~0，适配你的数据）
    plt.title('图1：所有原始点（X=时间，Y=频率，颜色=分贝）', fontsize=12)
    plt.xlabel('时间（秒）', fontsize=10)
    plt.ylabel('频率（Hz）', fontsize=10)
    # 颜色条标注（核心：颜色=分贝）
    cbar1 = plt.colorbar(scatter1)
    cbar1.set_label('分贝（dB）', fontsize=10)
    # 标注min_db=-70的分贝阈值（在颜色条上标线）
    cbar1.ax.axhline(y=min_db, color='black', linestyle='--', label=f'min_db={min_db}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('图1-所有原始点_时间-频率-分贝.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- 图2：仅被选中的锚点（X=时间，Y=频率，颜色=分贝） ----------
    plt.figure(figsize=(12, 6))
    # 锚点用大三角，颜色=分贝，更醒目
    scatter2 = plt.scatter(anchor_times, anchor_hzs, 
                           c=anchor_dbs, cmap='RdYlBu_r',
                           s=60, marker='^', alpha=0.9, vmin=-80, vmax=0)
    plt.title(f'图2：仅选中的锚点（共{len(anchor_times)}个）（X=时间，Y=频率，颜色=分贝）', fontsize=12)
    plt.xlabel('时间（秒）', fontsize=10)
    plt.ylabel('频率（Hz）', fontsize=10)
    # 颜色条
    cbar2 = plt.colorbar(scatter2)
    cbar2.set_label('分贝（dB）', fontsize=10)
    cbar2.ax.axhline(y=min_db, color='black', linestyle='--', label=f'min_db={min_db}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('图2-仅锚点_时间-频率-分贝.png', dpi=300, bbox_inches='tight')
    plt.show()

    # ---------- 可选：图3 - 所有点+锚点对比（最直观） ----------
    plt.figure(figsize=(12, 6))
    # 背景：所有原始点（浅色，低透明度）
    plt.scatter(filter_data_times, filter_data_hz, 
                c=filter_data_db, cmap='RdYlBu_r',
                s=10, alpha=0.3, vmin=-80, vmax=0, label=f'所有点（{len(filter_data_db)}个）')
    # 前景：锚点（大三角，高透明度）
    plt.scatter(anchor_times, anchor_hzs, 
                c=anchor_dbs, cmap='RdYlBu_r',
                s=60, marker='^', alpha=0.9, vmin=-80, vmax=0, label=f'锚点（{len(anchor_times)}个）')
    plt.title('图3：所有点+锚点对比（X=时间，Y=频率，颜色=分贝）', fontsize=12)
    plt.xlabel('时间（秒）')
    plt.ylabel('频率（Hz）')
    cbar3 = plt.colorbar()
    cbar3.set_label('分贝（dB）')
    cbar3.ax.axhline(y=min_db, color='black', linestyle='--')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig('图3-对比版_时间-频率-分贝.png', dpi=300)
    plt.show()

"""
    generate hashes from anchors
    save to dbfile

"""
def func6(anchors, song_name, db_path):
    save_data = {}
    for anchor in anchors:
        anchor_time, anchor_hz, target_hz, time_diff = anchor
        hash_str = f"{anchor_hz}|{target_hz}|{time_diff}"
        hash_value = hashlib.md5(hash_str.encode()).hexdigest()
        if save_data.get(hash_value):
            save_data[hash_value].append((anchor_time, song_name))       
        else:
            save_data.update({hash_value: [(anchor_time, song_name)]})
    
    if os.path.exists(db_path):
        if os.path.getsize(db_path) > 0:
            with open(db_path, 'r', encoding='utf-8') as f:
                db_data = json.load(f)
            for key in save_data:
                if db_data.get(key):
                    db_data[key].extend(save_data[key])
                else:
                    db_data.update({key: save_data[key]})
            save_data = db_data
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(anchors)} anchors to database.")
    return save_data




if __name__ == "__main__":
    db_path = "generate_V3_data.db"
    audio_path = "battle_music"
    with open(db_path, 'w', encoding='utf-8') as f:
        json.dump({}, f)

    for filename in os.listdir(audio_path):
        if not filename.lower().endswith(".wav"):
            continue
        file_path = os.path.join(audio_path, filename)
        rate, data , song_name = func1(file_path)
        data_magnitude, data_freq_bins_hz, data_time_bins_s = func2(rate, data)
        data_peaks_magnitude, data_peaks_hz, data_peak_times = func3(data_magnitude, data_freq_bins_hz, data_time_bins_s)
        filter_data_db, filter_data_hz, filter_data_times = func4(data_peaks_magnitude, data_peaks_hz, data_peak_times)
        anchors, filter_data_times, filter_data_hz, filter_data_db = func5(filter_data_db, filter_data_hz, filter_data_times)
        # plot1(anchors,filter_data_times, filter_data_hz, filter_data_db)
        save_data = func6(anchors, song_name, db_path)
