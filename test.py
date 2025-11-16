import os
import json
import numpy as np
import wfdb
from wfdb import processing
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from chronos import ChronosPipeline
import matplotlib.pyplot as plt


def extract_beats(sig, r_peaks):
    beats = []
    y = []
    x = []
    for i in range(len(r_peaks)-1):
        section = sig[r_peaks[i]:r_peaks[i+1]]
        y.append(sig[r_peaks[i]])
        x.append(r_peaks[i])
        beats.append(section)
    y.append(sig[r_peaks[-1]])
    x.append(r_peaks[-1])
    plt.figure()
    plt.plot(sig)
    plt.scatter(x,y)
    plt.show()

    return beats
fs = 500

data_folder = "014"

rec_name = "JS00412"
rec = wfdb.rdrecord(os.path.join(data_folder, rec_name))
sig = rec.p_signal[:, 1].astype(np.float32)

peaks = np.array(processing.xqrs_detect(sig=sig, fs=fs), dtype=int)

print(extract_beats(sig, peaks))

plt.figure()
plt.plot(sig)
plt.plot(extract_beats(sig, peaks))
plt.show()

