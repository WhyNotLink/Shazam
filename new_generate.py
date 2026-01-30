import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import butter, filtfilt
from scipy.signal.windows import hamming
from scipy.fft import fft
from scipy.signal import spectrogram
from scipy.io.wavfile import write

import sqlite3
import json
import hashlib
from datetime import datetime
import matplotlib.pyplot as plt

def process_audio_with_sliding_window(audio_file_path, window_size=4096, overlap_ratio=0.5):
    """
    使用滑动窗口技术处理音频，应用汉明窗函数和FFT变换
    
    参数:
        audio_file_path (str): 音频文件的路径
        window_size (int): 窗口大小，默认4096
        overlap_ratio (float): 重叠比例，默认0.5（50%重叠）
    
    返回:
        audio_data (ndarray): 处理后的音频数据
        sample_rate (int): 采样率
        f (ndarray): 频率数组
        t (ndarray): 时间帧数组
        Sxx (ndarray): 频谱图数据，形状为(频率数, 时间帧数)

        f\t	t0	t1	t2	...
        f0	Sxx[0,0]	Sxx[0,1]	Sxx[0,2]	...
        f1	Sxx[1,0]	Sxx[1,1]	Sxx[1,2]	...
        f2	Sxx[2,0]	Sxx[2,1]	Sxx[2,2]	...
        ...	...	...	...	...
    """
    
    sample_rate, audio_data = wavfile.read(audio_file_path)
    print(f"sample_rate:{sample_rate}")
    print(f"ndim:{audio_data.ndim}")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)


    # # FFT
    # audio = audio_data
    # N = len(audio)
    # fft_vals = np.fft.rfft(audio)  # 只取正频率
    # fft_freqs = np.fft.rfftfreq(N, 1/sample_rate)

    # # 找主频
    # magnitude = np.abs(fft_vals)
    # main_freq_idx = np.argmax(magnitude)
    # main_freq = fft_freqs[main_freq_idx]
    # print(f"主要频率: {main_freq:.2f} Hz")

    # # 绘图
    # plt.figure(figsize=(12, 4))
    # plt.plot(fft_freqs, magnitude)
    # plt.xlim(0, 6000)
    # plt.xlabel("Frequency (Hz)")
    # plt.ylabel("Amplitude")
    # plt.title("FFT Spectrum")

    low_freq = 20  # Hz
    high_freq = 20000  # Hz
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(4, [low, high], btype='band')
    audio_data = filtfilt(b, a, audio_data)

    f, t, Sxx = spectrogram(audio_data, fs=sample_rate, nperseg=1024, noverlap=512,window='hamming', mode='magnitude')
    audio_data = audio_data / np.max(np.abs(audio_data))

#    # 波形图
#     plt.figure(figsize=(12, 4))
#     plt.plot(np.arange(len(audio_data)) / sample_rate, audio_data)
#     plt.xlabel("Time [s]")
#     plt.ylabel("Amplitude")
#     plt.title("Waveform")

#     # 频谱图
#     plt.figure(figsize=(12, 6))
#     plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
#     plt.ylabel("Frequency [Hz]")
#     plt.xlabel("Time [s]")
#     plt.title("Spectrogram")
#     plt.colorbar(label='dB')
#     plt.show()
    
    # audio_to_save = (audio_data / np.max(np.abs(audio_data)) * 32767).astype(np.int16)
    # write("filtered_audio.wav", sample_rate, audio_to_save)
    # print("滤波后的音频已保存为 filtered_audio.wav")
    print(f"t: {t}")
    print(f"Sxx shape: {Sxx.shape}")

    return audio_data, sample_rate, f, t, Sxx


def compute_log_band_energy(f, t, Sxx, low_freq=20, high_freq=20000, num_bands=8):
    """    
    参数:
        frequencies: process_audio_with_sliding_window 返回的频率数组 f
        Sxx: process_audio_with_sliding_window 返回的幅度谱矩阵 (频率数 x 时间帧数)
        low_freq, high_freq: 对数频带范围
        num_bands: 对数频带数量
        
    返回:
        band_energy: 每个频带在每个时间帧的能量 (num_bands x time_frames)
        freq_edges: 每个频带上下限
    """
    actual_low = f[f > 0].min() 
    log_edges = np.linspace(np.log10(actual_low), np.log10(high_freq), num_bands + 1)
    freq_edges = 10 ** log_edges

    band_peak_energy = []
    band_peak_frequency = []
    filtered_peak_frequency = []
    filtered_peak_time = [] 
    num_peaks = 1
    for i in range(num_bands):
        mask = (f >= freq_edges[i]) & (f < freq_edges[i + 1])
        S_band = Sxx[mask, :] 
        peaks_idx = np.argsort(S_band, axis=0)[-num_peaks:, :]
        f_selected = f[mask]
        f_selected = f_selected[peaks_idx[0,:]]
        peaks_values = np.take_along_axis(S_band, peaks_idx, axis=0)
        band_peak_energy.append(peaks_values[0])
        band_peak_frequency.append(f_selected)
    band_peak_energy = np.array(band_peak_energy)
    band_peak_frequency = np.array(band_peak_frequency)
    average_peak_energy = np.mean(band_peak_energy, axis=0)
    keep_band_peak_frequency_idx = band_peak_energy >= average_peak_energy[np.newaxis,:]
    # keep_band_peak_frequency_idx = np.ones_like(band_peak_energy, dtype=bool)

    for i in range(band_peak_energy.shape[0]):
        filtered_peak_frequency.append(band_peak_frequency[i, keep_band_peak_frequency_idx[i,:]])
        filtered_peak_time.append(np.where(keep_band_peak_frequency_idx[i,:])[0])
        
    # band_energy = []
    # for i in range(num_bands):
    #     mask = (f >= freq_edges[i]) & (f < freq_edges[i + 1])
    #     energy = np.sum(Sxx[mask, :], axis=0)  
    #     band_energy.append(energy)

    # band_energy = np.array(band_energy)  
    # return band_energy, freq_edges

    # # 提取所有峰值的时间和频率
    # all_freqs = []
    # all_times = []
    # for freq_arr, idx_arr in zip(filtered_peak_frequency, filtered_peak_time):
    #     if len(freq_arr) > 0 and len(idx_arr) > 0:
    #         all_freqs.extend(freq_arr)
    #         all_times.extend(t[idx_arr])

    # # 绘制锚点图
    # plt.figure(figsize=(12, 6))
    # plt.scatter(all_times, all_freqs, s=15, c='red', alpha=0.7)  # s是点的大小，c是颜色
    # plt.xlabel('Time (s)', fontsize=12)
    # plt.ylabel('Frequency (Hz)', fontsize=12)
    # plt.title('Peak Frequency Points (Flat Time Axis)', fontsize=14)
    # plt.grid(True, alpha=0.3)
    # plt.yscale('log')  # 频率轴用对数刻度（和示例图一致）
    # plt.ylim(20, 20000)  # 限定频率范围
    # plt.show()
    # 提取所有峰值的时间和频率
    all_freqs = []
    all_times = []
    for freq_arr, idx_arr in zip(filtered_peak_frequency, filtered_peak_time):
        if len(freq_arr) > 0 and len(idx_arr) > 0:
            all_freqs.extend(freq_arr)
            all_times.extend(t[idx_arr])

    # ========== 新增：筛选最近的10个时间片段 ==========
    # 1. 找到t数组中最后10个时间点的起始值（即“最近10个片段”的时间起点）
    if len(t) >= 10:
        # 取t数组最后10个元素的最小值，作为筛选阈值
        recent_time_threshold = t[-10]  # t[-10]是倒数第10个时间点
    else:
        # 如果t数组长度不足10，显示全部
        recent_time_threshold = t[0]

    # 2. 筛选出时间≥阈值的峰值（只保留最近10个片段的峰值）
    recent_freqs = []
    recent_times = []
    for freq, time in zip(all_freqs, all_times):
        if time >= recent_time_threshold:
            recent_freqs.append(freq)
            recent_times.append(time)

    # 绘制锚点图（只显示最近10个片段）
    plt.figure(figsize=(12, 6))
    if len(recent_freqs) > 0:
        # 绘制最近10个片段的峰值
        plt.scatter(recent_times, recent_freqs, s=15, c='red', alpha=0.7)
        # 可选：标注“最近10个片段”
        plt.text(0.02, 0.98, '仅显示最近10个时间片段', 
                transform=plt.gca().transAxes, ha='left', va='top', 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.5, 0.5, '最近10个片段无有效峰值数据', ha='center', va='center', transform=plt.gca().transAxes)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Frequency (Hz)', fontsize=12)
    plt.title('Peak Frequency Points (Latest 10 Time Segments)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 频率轴用对数刻度
    plt.ylim(20, 20000)  # 限定频率范围
    # 可选：x轴聚焦最近10个片段的时间范围（更直观）
    plt.xlim(recent_time_threshold, t[-1])
    plt.show()

    return band_peak_energy, band_peak_frequency, filtered_peak_frequency, filtered_peak_time

def hashi_generate(filtered_peak_frequency,filtered_peak_time):
    pass    





if __name__ == "__main__":
    audio_path = "battle_music/battle_daoqi2.wav" 
    audio_data, sample_rate, frequencies, time_frames, spectrogram = process_audio_with_sliding_window(audio_path)
    compute_log_band_energy(frequencies, time_frames, spectrogram)
    # band_energy, freq_edges = compute_log_band_energy(frequencies, spectrogram)
    # print("频带能量:", band_energy)