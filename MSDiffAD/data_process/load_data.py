# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 09:50:08 2023

@author: Administrator
"""
import numpy as np
from tqdm import tqdm
from matchms.importing import load_from_msp
from matchms.importing import load_from_mgf

import pickle
from matchms.filtering import default_filters
from matchms.filtering import add_parent_mass, derive_adduct_from_name
from matchms.filtering import harmonize_undefined_inchikey, harmonize_undefined_inchi
from matchms.filtering import harmonize_undefined_smiles
from matchms.filtering import repair_inchi_inchikey_smiles
from matchms.filtering import derive_inchi_from_smiles, derive_smiles_from_inchi
from matchms.filtering import derive_inchikey_from_inchi
from matchms.filtering import select_by_mz
from matchms.filtering import normalize_intensities
from matchms.filtering import require_minimum_number_of_peaks

def CountAnnotations(spectra):
    '''
    Counting the number of spectrums and unique SMILES and inchikey
    '''
    inchi_lst = []
    smiles_lst = []
    inchikey_lst = []
    for i, spec in tqdm(enumerate(spectra)):
        inchi_lst.append(spec.get("inchi"))
        smiles_lst.append(spec.get("smiles"))
        inchikey = spec.get("inchikey")
        if inchikey is None:
            inchikey = spec.get("inchikey_inchi")
        inchikey_lst.append(inchikey)

    inchi_count = sum([1 for x in inchi_lst if x])
    smiles_count = sum([1 for x in smiles_lst if x])
    inchikey_count = sum([1 for x in inchikey_lst if x])
    print("Inchis:", inchi_count, "--", len(set(inchi_lst)), "unique")
    print("Smiles:", smiles_count, "--", len(set(smiles_lst)), "unique")
    print("Inchikeys:", inchikey_count, "--",
          len(set([x[:14] for x in inchikey_lst if x])), "unique (first 14 characters)")

def ApplyFilters(s):
    '''
    Basic matchs filters
    '''
    s = default_filters(s)
    s = derive_adduct_from_name(s)
    s = add_parent_mass(s, estimate_from_adduct=True)
    return s

def CleanMetadata(s):
    '''
    Clean (and extend) metadata
    '''
    s = harmonize_undefined_inchikey(s)
    s = harmonize_undefined_inchi(s)
    s = harmonize_undefined_smiles(s)
    s = repair_inchi_inchikey_smiles(s)
    return s

def CleanMetadata2(s):
    '''
    Convert entries where possible
    '''
    s = derive_inchi_from_smiles(s)
    s = derive_smiles_from_inchi(s)
    s = derive_inchikey_from_inchi(s)
    return s

def MinimalProcessing(spectrum):
    '''
    Filter spectrums by rules
    '''
    spectrum = normalize_intensities(spectrum)
    spectrum = select_by_mz(spectrum, mz_from=10.0, mz_to=1000.0)
    spectrum = require_minimum_number_of_peaks(spectrum, n_required=5)
    return spectrum

def CountFormulas(spectrums):
    '''
    Counting the number of formulas
    '''
    formulas = []
    name_to_formulas = []
    for spec in tqdm(spectrums):
        if spec.get("formula"):
            formulas.append(spec.get("formula"))
            name_to_formulas.append(spec.get("compound_name") + "---" + spec.get("formula"))
    return len(formulas),len(list(set(formulas)))

def SeparateSpec(spectrums):
    '''
    Separating datasets through positive and negative ion modes
    '''
    spectrums_positive = []
    spectrums_negative = []
    for i, spec in enumerate(spectrums):
        if spec.get("ionmode") == "positive":
            spectrums_positive.append(spec)
        elif spec.get("ionmode") == "negative":
            spectrums_negative.append(spec)
        else:
            print(f"No ionmode found for spectrum {i} ({spec.get('ionmode')})")
    return spectrums_positive,spectrums_negative

def Annotated(spectrums_pos_processing):
    '''
    Obtain annotated spectrums
    '''
    spectrums_pos_annotated = []
    for spec in tqdm(spectrums_pos_processing):
        inchikey = spec.get("inchikey")
        if inchikey is not None and len(inchikey)>13:
            if spec.get("smiles") or spec.get("inchi"):
                spectrums_pos_annotated.append(spec)
    return spectrums_pos_annotated

def PrecursorFilter(spectrums_pos_annotated):
    '''
    Using precursor ions to filtr spectrums
    '''
    spectrums_filter = []
    for spec in tqdm(spectrums_pos_annotated):
        precursor = spec.get('precursor_mz')
        if precursor == None:
            continue
        if 0 < float(precursor) < 999:
            spectrums_filter.append(spec)
    return spectrums_filter

def InstrumentFilter2(spectrums_pos_annotated):
    '''
    Separating datasets through instrument types
    '''
    orbitrap = []
    qtof = []
    other = []
    for spec in tqdm(spectrums_pos_annotated):
        # instrument = spec.metadata['ms_mass_analyzer']
        instrument = spec.get('source_instrument')
        # instrument = instrument.lower()
        # if instrument == 'orbitrap':
        #     orbitrap.append(spec)
        # elif 'qtof' in instrument or 'q-tof' in instrument or 'tof'in instrument:
        #     qtof.append(spec)
        # else:
        #     other.append(spec)
        if instrument == 'ESI-Orbitrap' or instrument == 'LC-ESI-Orbitrap':
            orbitrap.append(spec)
        elif instrument == 'ESI-qTof' or instrument == 'LC-ESI-qTof':
            qtof.append(spec)
        else:
            other.append(spec)
    return orbitrap, qtof, other


def InstrumentFilter(spectrums_pos_annotated):
    '''
    Separating datasets through instrument types
    '''
    orbitrap = []
    qtof = []
    other = []
    for spec in tqdm(spectrums_pos_annotated):
        instrument = spec.metadata['ms_mass_analyzer']
        # instrument = spec.get('source_instrument')
        instrument = instrument.lower()
        if instrument == 'orbitrap':
            orbitrap.append(spec)
        elif 'qtof' in instrument or 'q-tof' in instrument or 'tof'in instrument:
            qtof.append(spec)
        else:
            other.append(spec)
    return orbitrap, qtof, other


def MakeDataset(spectrums_pos_annotated,n_max=100,test_size=0.2,n_decimals=2):
    '''
    Divide the dataset into training and test data
    '''
    smiles_unique = []
    train_ref = []
    train_query = []
    test_ref = []
    test_query = []
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        smiles_unique.append(smi)
    smiles_unique = list(set(smiles_unique))
    spectrum_all = [[] for i in range(len(smiles_unique))]
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        position = smiles_unique.index(smi)
        spectrum_all[position].append(spec)
    number = int(len(spectrum_all)*(1-test_size))
    train_ref.append(spectrum_all[0])
    for spec_list in tqdm(spectrum_all[1:number]):
        if len(spec_list)  == 1:
            train_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            train_query.append([spec_list[p]])
            spec_list.pop(p)
            train_ref.append(spec_list)
    for spec_list in tqdm(spectrum_all[number:]):
        if len(spec_list)  == 1:
            test_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            test_query.append([spec_list[p]])
            spec_list.pop(p)
            test_ref.append(spec_list)
    return train_ref,train_query,test_ref,test_query

def MakeDataset2(spectrums_pos_annotated,n_max=100,test_size=0.2,n_decimals=2):
    '''
    Divide the dataset into training and test data
    '''
    smiles_unique = []
    train_ref = []
    train_query = []
    test_ref = []
    test_query = []
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        smiles_unique.append(smi)
    smiles_unique = list(set(smiles_unique))
    spectrum_all = [[] for i in range(len(smiles_unique))]
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        position = smiles_unique.index(smi)
        spectrum_all[position].append(spec)
    number = int(len(spectrum_all)*(1-test_size))

    train_ref.append(spectrum_all[0])
    train = spectrum_all[:number]

    number2 = int(len(spectrum_all) * test_size)
    print(len(spectrum_all))
    # print(number)
    # print(number2)
    # print(len(train))
    for spec_list in tqdm(spectrum_all[1:number2]):
        if len(spec_list)  == 1:
            train_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            train_query.append([spec_list[p]])
            spec_list.pop(p)
            train_ref.append(spec_list)
    for spec_list in tqdm(spectrum_all[number:]):
        if len(spec_list)  == 1:
            test_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            test_query.append([spec_list[p]])
            spec_list.pop(p)
            test_ref.append(spec_list)
    return train,train_ref,train_query,test_ref,test_query


from rdkit import Chem
from tqdm import tqdm


def get_canonical_smiles_from_spectrums(spectrum_list, tqdm_desc="规范化SMILES"):
    """
    1. 输入质谱列表
    2. 遍历并规范化SMILES
    3. 过滤掉无效或无法解析的SMILES
    4. 返回规范化后的SMILES列表
    """
    canonical_smiles_list = []
    failed_count = 0

    # 使用 tqdm 显示进度
    for spec in tqdm(spectrum_list, desc=tqdm_desc):
        smi = spec.get("smiles")

        if not smi:
            failed_count += 1
            continue

        try:
            # --- 科学规范化流程 ---
            # 1. 解析原始SMILES
            mol = Chem.MolFromSmiles(smi)

            if mol is not None:
                # 2. 生成规范化(Canonical)且包含立体信息的SMILES
                # 这步能确保：同物异码（如你之前的0.79情况）被统一
                can_smi = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                canonical_smiles_list.append(can_smi)
            else:
                failed_count += 1
        except Exception:
            failed_count += 1
            continue

    print(f"\n处理完成！")
    print(f"成功加入新列表: {len(canonical_smiles_list)} 条")
    print(f"解析失败并剔除: {failed_count} 条")

    return canonical_smiles_list




def MakeDataset_test(spectrums_pos_annotated,n_max=100,test_size=0.2,n_decimals=2):
    '''
    Divide the dataset into training and test data
    '''
    smiles_unique = []
    train_ref = []
    train_query = []
    test_ref = []
    test_query = []
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        smiles_unique.append(smi)
    smiles_unique = list(set(smiles_unique))
    spectrum_all = [[] for i in range(len(smiles_unique))]
    for spec in tqdm(spectrums_pos_annotated):
        smi = spec.get("smiles")
        position = smiles_unique.index(smi)
        spectrum_all[position].append(spec)
    number = int(len(spectrum_all)*(1-test_size))

    train_ref.append(spectrum_all[0])
    # train = spectrum_all[:number]
    #
    # number2 = int(len(spectrum_all) * test_size)
    # print(len(spectrum_all))
    # print(number)
    # print(number2)
    # print(len(train))
    for spec_list in tqdm(spectrum_all[1:]):
        if len(spec_list)  == 1:
            train_ref.append(spec_list)
        if len(spec_list) > 1:
            p = np.random.choice(len(spec_list))
            train_query.append([spec_list[p]])
            spec_list.pop(p)
            train_ref.append(spec_list)
    return train_ref,train_query


# def ProDataset(data,n_decimals,n_max):
#     '''
#     Obtain information about the spectrum
#     '''
#     data = [s for s1 in data for s in s1]
#     data_info = []
#     for spec in tqdm(data):
#         info = []
#         peaks = spec.peaks.to_numpy
#         if peaks.shape[0] > n_max:
#             n_large = np.argsort(peaks[:,1])[::-1][0:n_max]
#             n_large = sorted(n_large)
#             peaks = peaks[n_large,:]
#         smile = spec.get('smiles')
#         precursor = spec.metadata['precursor_mz']
#         mz = peaks[:,0]
#         mz = [str(round(i,n_decimals)) for i in mz]
#         inten = peaks[:,1]
#         info = []
#         info.append(smile)
#         info.append(precursor)
#         info.append([mz,inten])
#         data_info.append(info)
#     return data_info

def ProDataset(data,n_decimals,n_max):
    '''
    Obtain information about the spectrum
    '''
    data = [s for s1 in data for s in s1]
    data_info = []
    for spec in tqdm(data):
        info = []
        peaks = spec.peaks.to_numpy
        if peaks.shape[0] > n_max:
            n_large = np.argsort(peaks[:,1])[::-1][0:n_max]
            n_large = sorted(n_large)
            peaks = peaks[n_large,:]
        smile = spec.get('smiles')
        precursor = spec.metadata['precursor_mz']
        parent_mass = spec.metadata['parent_mass']
        mz = peaks[:,0]

        maxlen = 100
        peaks_re = [round((float(i) * 100) % 1, 2) for i in mz]
        precursor_res = round((float(parent_mass) * 100) % 1, 2)
        res = [precursor_res] + peaks_re
        if len(res) < maxlen:
            # 长度不足，补 0.0
            res.extend([0.0] * (maxlen - len(res)))
        else:
            # 长度超出，截断（防止维度溢出）
            res = res[: maxlen]

        mz = [str(round(i,n_decimals)) for i in mz]
        # # mz_residuals
        inten = peaks[:,1]
        info = []
        info.append(smile)
        info.append(precursor)
        info.append([mz,inten,res])
        data_info.append(info)
    return data_info



import json
from collections import Counter
from tqdm import tqdm

def count_instruments_and_save(spectrums_pos_annotated, output_json_path):
    """
    统计 'source_instrument' 类型及数量，并保存到 JSON 文件。

    Parameters
    - spectrums_pos_annotated: 可迭代对象，元素为 dict 或具备 .get 方法的对象
    - output_json_path: 输出 JSON 文件路径
    """
    counter = Counter()

    for spec in tqdm(spectrums_pos_annotated, desc="Counting instruments"):
        instrument = spec.get('source_instrument')
        # # 可选清洗：统一大小写、去除两端空格
        # if instrument is None:
        #     continue
        # if isinstance(instrument, str):
        #     instrument_clean = instrument.strip()
        #     if instrument_clean == "":
        #         continue
        #     # 视需求选择大小写统一：例如全部转为小写
        #     instrument_clean = instrument_clean.lower()
        # else:
        #     # 非字符串的情况，转为字符串
        #     instrument_clean = str(instrument)

        counter[instrument] += 1

    # 转为普通字典
    result = dict(counter)

    # 保存为 JSON
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result

from rdkit import Chem
import copy
def get_canonical_spectrums(spectrums):
    """
    针对质谱对象列表进行 SMILES 规范化
    """
    canonical_list = []

    for spec in spectrums:
        # 兼容性处理：尝试作为类属性或字典键获取
        if hasattr(spec, 'get'):
            smile = spec.get('smiles')
        else:
            continue

        if not smile or not isinstance(smile, str):
            continue

        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                canonical_smile = Chem.MolToSmiles(mol, isomericSmiles=True)

                # --- 修正部分 ---
                # 建议克隆对象，避免原地修改带来的副作用
                new_spec = copy.deepcopy(spec)

                # 如果是 matchms 对象，应更新其 metadata
                if hasattr(new_spec, 'metadata'):
                    new_spec.metadata['smiles'] = canonical_smile
                else:
                    # 如果是普通字典
                    new_spec['smiles'] = canonical_smile

                canonical_list.append(new_spec)
        except:
            continue

    return canonical_list


if __name__ == '__main__':
    # # 定义输入的 GNPS 质谱数据文件（MSP 格式）的绝对路径
    # gnps_file = 'E:/MS_Data/MSBERT/ALL_GNPS.msp'
    #
    # # 步骤 1：从 MSP 文件中加载原始质谱数据，并将其转换为列表格式
    # spectrums = list(load_from_msp(gnps_file))
    #
    # # 统计并打印当前原始质谱数据中的标注信息（如注释数量等）
    # CountAnnotations(spectrums)
    #
    # # 步骤 2：对列表中的每个质谱对象进行过滤处理（例如：去除无效峰、噪声等）
    # spectrums = [ApplyFilters(s) for s in spectrums]
    #
    # # 步骤 3：统计过滤后质谱数据中的化学式总数以及去重后的唯一化学式数量
    # number_formula, fornuma_unique = CountFormulas(spectrums)
    #
    # # 步骤 4：对质谱的元数据（Metadata）进行第一轮清洗（配合 tqdm 显示进度条）
    # spectrums = [CleanMetadata(s) for s in tqdm(spectrums)]
    #
    # # 步骤 5：对质谱的元数据进行第二轮深度清洗（配合 tqdm 显示进度条）
    # spectrums = [CleanMetadata2(s) for s in tqdm(spectrums)]
    #
    # # 再次统计并打印清洗完成后的质谱数据标注信息
    # CountAnnotations(spectrums)
    #
    # # 步骤 6：根据电离模式，将质谱数据分离为正离子模式（Positive）和负离子模式（Negative）
    # spectrums_positive, spectrums_negative = SeparateSpec(spectrums)
    #
    # # 统计并打印分离出来的正离子质谱数据的标注信息
    # CountAnnotations(spectrums_positive)
    #
    # # 步骤 7：计算每个正离子质谱的碎片峰数量，并统计/打印其中峰数量小于 5 个的质谱个数
    # number_of_peaks = np.array([len(s.peaks) for s in spectrums_positive])
    # print(f"{np.sum(number_of_peaks < 5)} spectra have < 5 peaks")
    #
    # # 步骤 8：对正离子质谱进行最小化预处理（例如：基本的强度归一化或低丰度峰剔除）
    # spectrums_pos_processing = [MinimalProcessing(s) for s in spectrums_positive]
    #
    # # 步骤 9：过滤掉预处理后变为 None 的无效质谱数据，只保留有效对象
    # spectrums_pos_processing = [s for s in spectrums_pos_processing if s is not None]
    #
    # # 统计并打印预处理完的正离子质谱标注信息
    # CountAnnotations(spectrums_pos_processing)
    #
    # # 步骤 10：对处理后的正离子质谱进行结构或化学信息标注
    # spectrums_pos_annotated = Annotated(spectrums_pos_processing)
    #
    # # 打印标注后的正离子质谱总数量
    # print(len(spectrums_pos_annotated))
    #
    # # （已注释）获取规范化的标准质谱数据
    # # spectrums_pos_annotated = get_canonical_spectrums(spectrums_pos_annotated)
    #
    # # 再次打印确认数量（确保与上一句输出一致）
    # print(len(spectrums_pos_annotated))
    #
    # # 步骤 11：将标注好的完整正离子质谱数据集保存为 pickle 二进制文件
    # pickle.dump(spectrums_pos_annotated, open('D:/Projects/MSDiffAD/data/gnps_spectrums.pickle', 'wb'))
    #
    # # 统计并打印最终保存的正离子质谱数据的标注信息
    # CountAnnotations(spectrums_pos_annotated)

    # # 步骤 12：重新读取刚刚保存的 pickle 文件，验证数据是否能正常反序列化加载
    # with open('D:/Projects/MSDiffAD/data/gnps_spectrums.pickle', 'rb') as f:
    #     spectrums_pos_annotated = pickle.load(f)

    # 步骤 12：重新读取刚刚保存的 pickle 文件，验证数据是否能正常反序列化加载
    with open('D:/Projects/MSBERT/data/GNPSdata_f1/spectrums_pos_annotated.pickle', 'rb') as f:
        spectrums_pos_annotated = pickle.load(f)
    spectrums_pos_annotated = spectrums_pos_annotated[:2000]
    # 打印成功读取出来的质谱数据数量
    print(len(spectrums_pos_annotated))

    # 步骤 13：基于前体离子（Precursor）的相关指标对正离子质谱进行进一步筛选
    spectrums_filetr = PrecursorFilter(spectrums_pos_annotated)

    # 步骤 14：根据质谱仪器的类型（Orbitrap、Q-TOF、其他）对筛选后的数据进行分类
    orbitrap, qtof, other = InstrumentFilter(spectrums_filetr)

    # 打印属于 Orbitrap（静电场轨道阱）类型的质谱数量
    print(len(orbitrap))

    # 步骤 15：将 Orbitrap 数据划分为训练集和测试集（并切分为参考集 Ref 和查询集 Query）
    # 参数设置：单个化合物最大质谱数 n_max=99，测试集比例 20%，保留 2 位小数
    ob_train_ref, ob_train_query, ob_test_ref, ob_test_query = MakeDataset(orbitrap, n_max=99, test_size=0.2,
                                                                           n_decimals=2)

    # 步骤 16：分别将 Orbitrap 的训练参考集、训练查询集、测试参考集、测试查询集保存为 pickle 文件
    pickle.dump(ob_train_ref, open('D:/Projects/MSDiffAD/data/ob_train_ref.pickle', 'wb'))
    pickle.dump(ob_train_query, open('D:/Projects/MSDiffAD/data/ob_train_query.pickle', 'wb'))
    pickle.dump(ob_test_ref, open('D:/Projects/MSDiffAD/data/ob_test_ref.pickle', 'wb'))
    pickle.dump(ob_test_query, open('D:/Projects/MSDiffAD/data/ob_test_query.pickle', 'wb'))

    # 步骤 17：分别将 Q-TOF 类型和其他类型的质谱数据保存为独立的 pickle 文件
    pickle.dump(qtof, open('D:/Projects/MSDiffAD/data/qtof.pickle', 'wb'))
    pickle.dump(other, open('D:/Projects/MSDiffAD/data/other.pickle', 'wb'))