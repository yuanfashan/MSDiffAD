# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:50:48 2023

@author: Administrator
"""
import numpy as np
import torch

def MakeTestData(sentences,precursor,word2idx,maxlen):
    '''
    Build train test set
    '''
    intensity = [i[1] for i in sentences]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i) for i in j] for j in peaks]
    peaks = [["%.2f"%(i) for i in j] for j in peaks]
    precursor = ["%.2f"%(i) for i in precursor]
    token_list = []
    for p in range(len(peaks)):
        arr = [word2idx[s] for s in peaks[p]]
        arr = [word2idx[precursor[p]]] + arr
        token_list.append(arr)
    test_data = []
    for i in range(len(token_list)):
        input_ids = token_list[i]
        n_pad = maxlen - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)
        intensity2 = intensity[i]/max(intensity[i])
        intensity2 = np.hstack((intensity2,np.zeros(maxlen-len(intensity2))))
        intensity2 = intensity2.reshape(1,len(intensity2))
        test_data.append([input_ids,intensity2])
    return test_data

def MakeTrainData(sentences,parent_mass, maxlen):
    '''
    Build train data set
    '''
    intensity = [i[1] for i in sentences]
    res = [i[2] for i in sentences]
    res = np.array(res, dtype=np.float32)
    # res = [[float(i) for i in j] for j in res]
    intensity = [np.hstack((2,i)) for i in intensity]
    peaks = [i[0] for i in sentences]
    peaks = [[float(i) for i in j] for j in peaks]

    peakss = [
        [999.99 if s > 999.99 else s for s in row]
        for row in peaks
    ]

    # peaks = [["%.2f"%(i) for i in j] for j in peaks]
    #
    #
    # precursor = ["%.2f"%float(i) for i in precursor]
    peaks = [["%.2f"%(i) for i in j] for j in peakss]
    precursor = ["%.2f" % float(i) for i in parent_mass]


    word_list = list(np.round(np.linspace(0,1000,100*1000,endpoint=False),2))
    word_list = ["%.2f"%(i) for i in word_list]
    word2idx = {'[PAD]' : 0, '[MASK]' : 1}
    for i, w in enumerate(word_list):
        word2idx[w] = i + 2
    token_list = []
    for p in range(len(peaks)):
        arr = [word2idx[s] for s in peaks[p]]
        arr = [word2idx[precursor[p]]] + arr
        token_list.append(arr)
    train_data = []
    for i in range(len(token_list)):
        input_ids = token_list[i]
        n_pad = maxlen - len(input_ids)
        input_ids.extend([word2idx['[PAD]']] * n_pad)
        intensity2 = intensity[i]/max(intensity[i])
        intensity_bin = get_intensity_rank_labels(intensity2)
        intensity2 = np.hstack((intensity2,np.zeros(maxlen-len(intensity2))))
        intensity_bin = np.hstack((intensity_bin, np.zeros(maxlen - len(intensity_bin), dtype=int)))
        intensity2 = intensity2.reshape(1,len(intensity2))
        res2 = res[i]

        train_data.append([input_ids,intensity2,res2,intensity_bin])
    return train_data,word2idx


def get_intensity_rank_labels(intensity, num_bins=5):
    """
    1. 传入长度不固定的 intensity (list 或 ndarray)
    2. 进行排序，得到每个元素的百分比排名
    3. 划分为 0-4 五个等级
    """
    if len(intensity) == 0:
        return np.array([])
    ints = np.array(intensity, dtype=float)

    ranks = np.argsort(np.argsort(ints))

    labels = (ranks / len(ints) * num_bins).astype(int)

    labels = np.clip(labels, 0, num_bins - 1)

    return labels














