import generate_V3 as gv3
import time
import json
import pyaudio
import os
import numpy as np
import pydub
import librosa
import hashlib
import sounddevice as sd
from scipy.io.wavfile import write as wav_write


"""
    recording detected file

"""
def func1():
    filename = "V3_detecting.wav"
    duration = 5
    fs = 48000
    channels = 2
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
    sd.wait()
    wav_write(filename, fs, recording)
    print(f"音频录制完成，已保存为：{filename}")


"""
    use generate.py file to generate detected file fingerprint anchor

"""
def func2(wav_path):
    rate, data , song_name = gv3.func1(wav_path)
    data_magnitude, data_freq_bins_hz, data_time_bins_s = gv3.func2(rate, data)
    data_peaks_magnitude, data_peaks_hz, data_peak_times_s = gv3.func3(data_magnitude, data_freq_bins_hz, data_time_bins_s)
    filter_data_db, filter_data_hz, filter_data_times = gv3.func4(data_peaks_magnitude, data_peaks_hz, data_peak_times_s)
    anchor_pair, filter_data_times, filter_data_hz, filter_data_db = gv3.func5(filter_data_db, filter_data_hz, filter_data_times)
    return anchor_pair

"""
    score 1

"""
def func3(anchor_pair,hash_data):
    slave_data = {}
    slave_score = {} 
    for anchor in anchor_pair:
        anchor_time, anchor_hz, target_hz, time_diff = anchor
        hash_str = f"{anchor_hz}|{target_hz}|{time_diff}"
        hash_value = hashlib.md5(hash_str.encode()).hexdigest()

        if hash_value in hash_data:
            for item in hash_data[hash_value]:
                item_anchor_time , item_songname = item

                if item_songname in slave_data:
                    slave_data[item_songname].append((hash_value,item_anchor_time,anchor_time))
                else:
                    slave_data.update({item_songname:[(hash_value,item_anchor_time,anchor_time)]})
    print(len(slave_data))
    if slave_data:    
        for name in slave_data:
            score = 0
            benchmark_slave_time = slave_data[name][0][1]
            benchmark_host_time = slave_data[name][0][2]
            for i in range(len(slave_data[name])):
                if i !=0:
                    side_slave_time = slave_data[name][i][1]
                    side_host_time = slave_data[name][i][2]
                    offset_slave_time = abs(side_slave_time - benchmark_slave_time)
                    offset_host_time = abs(side_host_time - benchmark_host_time)
                    if abs(offset_host_time- offset_slave_time) <= 0.10000:
                        score +=1
            slave_score.update({name:score})

    sorted_scores = sorted(slave_score.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores)

"""
    score 2

"""
def func4(anchor_pair,hash_data):
    slave_data = {}
    slave_score = {} 
    for anchor in anchor_pair:
        anchor_time, anchor_hz, target_hz, time_diff = anchor
        hash_str = f"{anchor_hz}|{target_hz}|{time_diff}"
        hash_value = hashlib.md5(hash_str.encode()).hexdigest()
        if hash_value in hash_data:
            for item in hash_data[hash_value]:
                item_anchor_time , item_songname = item

                if item_songname in slave_data:
                    slave_data[item_songname].append((hash_value,item_anchor_time,anchor_time))
                else:
                    slave_data.update({item_songname:[(hash_value,item_anchor_time,anchor_time)]})
    print(len(slave_data))
    if slave_data:    
        for name in slave_data:
            offset_dic = {}
            for i in range(len(slave_data[name])):
                slave_time = slave_data[name][i][1]
                host_time = slave_data[name][i][2]
                # offset_time = abs(host_time-slave_time)
                # offset_time = round(offset_time,1)
                offset_time = round(slave_time-host_time,1)
                if offset_time in offset_dic:
                    offset_dic[offset_time] += 1
                else:
                    offset_dic.update({offset_time:1})
            max_offset = max(offset_dic.values())
            slave_score.update({name:max_offset})
    sorted_scores = sorted(slave_score.items(), key=lambda x: x[1], reverse=True)
    print(sorted_scores)





if __name__ == "__main__":
    db_path = "generate_V3_data.db"
    with open(db_path, "r", encoding="utf-8") as f:
        hash_data = json.load(f)
        
    while True:
        print("start")
        func1()
        anchor_pair = func2("V3_detecting.wav")
        func4(anchor_pair,hash_data)
        print("end")