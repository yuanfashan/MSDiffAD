# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 16:58:02 2023

@author: Administrator
"""
import torch
from data_process.dataset import MyDataSet, MyDataSet3
import torch.utils.data as Data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_process.load_data import ProDataset, MakeDataset
from data_process.process_data import MakeTestData, MakeTrainData
from utils.ms2vec import ms_to_vec
from matchms.importing import load_from_msp, load_from_mgf, load_from_mzml
from scipy.spatial.distance import cosine


# from draft.Ablation.MLM.model.a6.train import MyDataSet

def CalCosineTop(qtof_ref, qtof_query):
    '''
    Library matching performance for calculating cosine similarity
    '''
    qtof_ref_ = [i for s in qtof_ref for i in s]
    qtof_query_ = [i for s in qtof_query for i in s]
    qtof_ref_peak = [s.peaks.to_numpy for s in qtof_ref_]
    qtof_query_peak = [s.peaks.to_numpy for s in qtof_query_]
    m = ms_to_vec()
    qtof_ref_peak = [m.transform(i) for i in qtof_ref_peak]
    qtof_ref_peak = np.array(qtof_ref_peak)
    qtof_query_peak = [m.transform(i) for i in qtof_query_peak]
    qtof_query_peak = np.array(qtof_query_peak)
    qtof_ref = ProDataset(qtof_ref, 2, 99)
    qtof_query = ProDataset(qtof_query, 2, 99)
    smile_ref = [i[0] for i in qtof_ref]
    smile_query = [i[0] for i in qtof_query]
    cosinetop = SearchTop(qtof_ref_peak, qtof_query_peak, smile_ref, smile_query, batch=50)
    return cosinetop


def ParseOrbitrap(file):
    '''
    Parsing Orbitrap Datasets
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    # parent_mass = [i[3] for i in ref]
    return train_ref, msms, precursor, smiles


def ParseOrbitrap_test(file):
    '''
    Parsing Orbitrap Datasets
    '''
    with open(file, 'rb') as f:
        train_ref = pickle.load(f)
    # train_ref = train_ref[1:]
    ref = ProDataset(train_ref, 2, 99)
    msms = [i[2] for i in ref]
    precursor = [i[1] for i in ref]
    smiles = [i[0] for i in ref]
    return train_ref, msms, precursor, smiles


def CalMSBERTTop(MSBERTModel, ref_data, query_data, smile_ref, smile_query):
    '''
    Library matching performance for MSBERT
    '''
    ref_arr = ModelEmbed(MSBERTModel, ref_data, 64)
    query_arr = ModelEmbed(MSBERTModel, query_data, 64)
    top = SearchTop(ref_arr, query_arr, smile_ref, smile_query, batch=50)
    return top


def ParseOtherData(other):
    '''
    Parsing other types of instrument datasets
    '''
    other_ref, other_query, _, _ = MakeDataset(other, n_max=99, test_size=0, n_decimals=2)
    other_ref = ProDataset(other_ref, 2, 99)
    other_query = ProDataset(other_query, 2, 99)
    msms_ref = [i[2] for i in other_ref]
    msms_query = [i[2] for i in other_query]
    smile_ref = [i[0] for i in other_ref]
    smile_query = [i[0] for i in other_query]
    precursor_ref = [i[1] for i in other_ref]
    precursor_query = [i[1] for i in other_query]
    ref_data, word2idx = MakeTrainData(msms_ref, precursor_ref, 100)
    query_data, word2idx = MakeTrainData(msms_query, precursor_query, 100)
    return ref_data, query_data, smile_ref, smile_query


def DatasetSep(input_ids, intensity, val_size=0.1):
    '''
    Split the dataset for training and testing
    '''
    n = len(intensity)
    perm = np.random.permutation(n)
    n_train = int(n * (1 - val_size))
    perm_train = perm[0:n_train]
    perm_val = perm[n_train:]
    input_ids_train = [input_ids[x] for x in perm_train]
    input_ids_val = [input_ids[x] for x in perm_val]
    intensity_train = [intensity[x] for x in perm_train]
    intensity_val = [intensity[x] for x in perm_val]
    return input_ids_train, intensity_train, input_ids_val, intensity_val


def ModelEmbed(model, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity, res, inten_bin = zip(*test_data)
    input_ids = [torch.LongTensor(i) for i in input_ids]
    intensity = [torch.FloatTensor(i) for i in intensity]
    res = [torch.FloatTensor(i) for i in res]
    inten_bin = [torch.LongTensor(i) for i in inten_bin]
    dataset = MyDataSet3(input_ids, intensity, res)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_, res) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            res = res.to(device)
            # pool = model.predict(input_id,intensity_,res)
            # msbert
            pool = model.predict(input_id, intensity_)

            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def ModelEmbed_projection(msencoder, projection, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity = zip(*test_data)
    intensity = [torch.FloatTensor(i) for i in intensity]
    input_ids = [torch.LongTensor(i) for i in input_ids]
    dataset = MyDataSet(input_ids, intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    msencoder = msencoder.to(device)
    msencoder.eval()
    projection = projection.to(device)
    projection.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            pool = msencoder.predict(input_id, intensity_)
            pool = projection(pool)
            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def ModelEmbed2(model, test_data, batch_size):
    '''
    Using MSBERT for MS/MS embedding
    Parameters
    ----------
    model
        MSBERT MSBertModel.
    test_data : list
        Data for MSBERT.
    batch_size : int

    Returns
    -------
    embed_arr : array
        MSBERT embedding results.

    '''
    input_ids, intensity = zip(*test_data)
    intensity = [torch.FloatTensor(i) for i in intensity]
    input_ids = [torch.LongTensor(i) for i in input_ids]
    dataset = MyDataSet(input_ids, intensity)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    embed_list = []
    with torch.no_grad():
        for step, (input_id, intensity_) in tqdm(enumerate(dataloader)):
            input_id = input_id.to(device)
            intensity_ = intensity_.to(device)
            pool = model.predict(input_id, intensity_)
            embed_list.append(pool.cpu().numpy())
    embed_arr = np.concatenate(embed_list)
    embed_arr = embed_arr.reshape(embed_arr.shape[0], embed_arr.shape[2])
    return embed_arr


def SearchTop(dataset_arr, query_arr, dataset_smiles, query_smiles, batch):
    '''
    Top-n metrics for computing library matching
    '''
    top1 = []
    top5 = []
    top10 = []
    start = 0
    n_dataset = np.linalg.norm(dataset_arr, axis=1)
    n_dataset = n_dataset.reshape(n_dataset.shape[0], 1)
    while start < query_arr.shape[0]:
        end = start + batch
        q_i = query_arr[start:end, :]
        n_q = np.linalg.norm(q_i, axis=1)
        n_q = n_q.reshape(1, n_q.shape[0])
        n_q = np.repeat(n_q, n_dataset.shape[0], axis=0)
        dot = np.dot(dataset_arr, q_i.T)
        n_d = np.repeat(n_dataset, q_i.shape[0], axis=1)
        sim = dot / (n_d * n_q)
        sort = np.argsort(sim, axis=0)
        sort = np.flipud(sort)
        for s in range(sort.shape[1]):
            smi_q = query_smiles[(s + start)]
            smi_dataset = [dataset_smiles[i] for i in sort[0:10, s]]
            if smi_q in smi_dataset:
                top10.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:5, s]]
            if smi_q in smi_dataset:
                top5.append(1)
            smi_dataset = [dataset_smiles[i] for i in sort[0:1, s]]
            if smi_q in smi_dataset:
                top1.append(1)
        start += batch
    top1 = len(top1) / len(query_smiles)
    top5 = len(top5) / len(query_smiles)
    top10 = len(top10) / len(query_smiles)
    return [top1, top5, top10]


import torch.nn.functional as F


def SearchTop_MemoryEfficient(dataset_arr, query_arr, dataset_smiles, query_smiles,
                              batch=500, dataset_chunk_size=5000, device='cuda'):
    """
    解决方案1: Dataset分块 + Query批处理
    适用于: Dataset太大，但Query可以批量处理
    """
    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.cuda.empty_cache()

    n_dataset = len(dataset_arr)
    n_queries = len(query_arr)

    # 为每个query存储最佳匹配
    query_best_scores = np.full((n_queries, 10), -np.inf, dtype=np.float32)
    query_best_indices = np.full((n_queries, 10), -1, dtype=np.int32)

    print(f"Processing {n_dataset} dataset samples in chunks of {dataset_chunk_size}")

    with torch.no_grad():
        # 分块处理dataset
        for dataset_start in range(0, n_dataset, dataset_chunk_size):
            dataset_end = min(dataset_start + dataset_chunk_size, n_dataset)
            dataset_chunk = torch.from_numpy(dataset_arr[dataset_start:dataset_end]).float().to(device)

            # print(
            #     f"Processing dataset chunk {dataset_start // dataset_chunk_size + 1}/{(n_dataset - 1) // dataset_chunk_size + 1}")

            # 批处理query
            for query_start in range(0, n_queries, batch):
                query_end = min(query_start + batch, n_queries)
                query_batch = torch.from_numpy(query_arr[query_start:query_end]).float().to(device)

                # 计算当前chunk的余弦相似度
                cosine_sim = F.cosine_similarity(
                    query_batch.unsqueeze(1),
                    dataset_chunk.unsqueeze(0),
                    dim=2
                )  # shape: (batch_size, chunk_size)

                # 更新每个query的最佳匹配
                for i in range(cosine_sim.shape[0]):
                    query_idx = query_start + i
                    current_scores = cosine_sim[i].cpu().numpy()
                    current_indices = np.arange(dataset_start, dataset_end)

                    # 合并当前分数和历史最佳分数
                    all_scores = np.concatenate([query_best_scores[query_idx], current_scores])
                    all_indices = np.concatenate([query_best_indices[query_idx], current_indices])

                    # 获取top-10
                    if len(all_scores) > 10:
                        top10_mask = np.argpartition(all_scores, -10)[-10:]
                        query_best_scores[query_idx] = all_scores[top10_mask]
                        query_best_indices[query_idx] = all_indices[top10_mask]

                        # 排序
                        sort_order = np.argsort(query_best_scores[query_idx])[::-1]
                        query_best_scores[query_idx] = query_best_scores[query_idx][sort_order]
                        query_best_indices[query_idx] = query_best_indices[query_idx][sort_order]

            # 清理GPU内存
            del dataset_chunk
            torch.cuda.empty_cache() if device.type == 'cuda' else None

    # 计算top-k准确率
    top1_count = 0
    top5_count = 0
    top10_count = 0

    for i in range(n_queries):
        query_smile = query_smiles[i]
        best_indices = query_best_indices[i]
        valid_indices = best_indices[best_indices >= 0]  # 过滤无效索引

        if len(valid_indices) > 0:
            # 检查命中
            if query_smile == dataset_smiles[valid_indices[0]]:
                top1_count += 1
            if query_smile in [dataset_smiles[idx] for idx in valid_indices[:5]]:
                top5_count += 1
            if query_smile in [dataset_smiles[idx] for idx in valid_indices[:10]]:
                top10_count += 1

    return [top1_count / n_queries, top5_count / n_queries, top10_count / n_queries]


def PlotStepLoss(train_loss, step=100):
    '''
    Draw loss curve
    '''
    all_loss = [p for i in train_loss for p in i]
    step_loss = [all_loss[i:i + step] for i in range(0, len(all_loss), step)]
    step_loss = [np.nanmean(i) for i in step_loss]
    plt.plot(step_loss)
    plt.xlabel('Steps')
    plt.ylabel('Loss')


def ProcessMSP(file):
    '''
    Load dataset from MSP and prepare for MSBERT
    Parameters
    ----------
    file : str
        The Path of MS/MS spectra.
    Returns
    -------
    data : list
        Data for MSBERT.
    smiles : TYPE
        SMILES for MS/MS.
    '''
    p = file.find('.')
    if file[p:] == '.msp':
        msms = list(load_from_msp(file))
    elif file[p:] == '.mgf':
        msms = list(load_from_mgf(file))
    elif file[p:] == '.mzML':
        msms = list(load_from_mzml(file))
    pro_data = ProDataset([msms], 2, 99)
    msms = [i[2] for i in pro_data]
    precursor = [i[1] for i in pro_data]
    smiles = [i[0] for i in pro_data]
    data, _ = MakeTrainData(msms, precursor, 100)
    return data, smiles


def MSBERTSimilarity(r_arr, q_arr):
    '''
    Parameters
    ----------
    r_arr : array
        Reference vectors Embedded by MSBERT.
    q_arr : array
        Query vectors Embedded by MSBERT.

    Returns
    -------
    similarity : arrray
        Similarity matrix.

    '''
    similarity = np.zeros((r_arr.shape[0], q_arr.shape[0]))
    for i in range(r_arr.shape[0]):
        v1 = r_arr[i, :]
        for j in range(q_arr.shape[0]):
            v2 = q_arr[j, :]
            cos_ij = cosine(v1, v2)
            similarity[i, j] = 1 - cos_ij
    return similarity

