# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:59:38 2023

@author: Administrator
"""
from data_process.process_data import MakeTrainData
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient, ModelEmbed2, SearchTop
import torch
from model.representation.representation_model import MSDiffusion



if __name__ == '__main__':
    model_file = 'D:/Projects/MSDiffAD/train/ms_repre_pth/msdiffad_encoder.pt'
    model = MSDiffusion(100002, 512, 6, 16, 0,100,3)
    model.load_state_dict(torch.load(model_file))

    max_len = 100
    ref, test_msms1, test_precursor1, ref_smiles = ParseOrbitrap(
        'D:/Projects/MSDiffAD/data/ob_test_ref.pickle')
    ref_data, _ = MakeTrainData(test_msms1, test_precursor1, max_len)
    ref_arr = ModelEmbed(model, ref_data, 64)

    query, query_msms1, query_precursor1, query_smiles = ParseOrbitrap(
        'D:/Projects/MSDiffAD/data/ob_test_query.pickle')
    query_data, _ = MakeTrainData(query_msms1, query_precursor1, max_len)
    query_arr = ModelEmbed(model, query_data, 64)

    top = SearchTop_MemoryEfficient(ref_arr, query_arr, ref_smiles, query_smiles)
    # top = SearchTop(ref_arr, query_arr, ref_smiles, query_smiles, batch=1000)
    print(top)


