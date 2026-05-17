from model.diffusion.generater import Trainer
from utils.diff_utils import set_seed
from utils.diff_config import get_config
import os
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient
from data_process.process_data import MakeTrainData
import random
import numpy as np
from torch.utils.data import Dataset



class MyDataSet(Dataset):
    def __init__(self, input_ids, intensity, mask_ratio=0.1):
        '''
        初始化数据集，接收两个输入参数：input_ids 和 intensity

        :param input_ids: 输入 ID，形状为 (样本数, 100)
        :param intensity: 强度值，形状为 (样本数, 100)
        :param mask_ratio: 用于随机掩蔽的位置比例（默认为 0.1）
        '''
        self.input_ids = input_ids  # 记录 input_ids
        self.intensity = intensity  # 记录 intensity
        self.mask_ratio = mask_ratio  # 掩蔽的比例

    def __len__(self):
        '''
        返回数据集的大小（即样本数）
        '''
        return len(self.input_ids)

    def __getitem__(self, idx):
        '''
        根据索引返回样本的 input_ids, intensity, attention_mask 和 masked_indices0，返回字典格式

        :param idx: 样本的索引
        :return: 一个字典，包含 'input_ids', 'intensity', 'attention_mask' 和 'masked_indices0'
        '''
        # 获取当前样本的 input_ids 和 intensity
        input_ids = torch.tensor(self.input_ids[idx], dtype=torch.long)
        # intensity = torch.tensor(self.intensity[idx], dtype=torch.float)
        intensity = self.intensity[idx].clone().detach().float()

        # 生成 attention_mask，input_id 为 0 时 mask 为 0.0，其他为 1.0
        attention_mask = (input_ids != 0).float()

        # 生成 masked_indices0，按 0.1 的比率随机标记非零位置为 1
        # 获取所有非零位置的索引
        non_zero_indices = (input_ids != 0).nonzero(as_tuple=True)[0]

        # 随机选择 10% 的非零位置
        # num_masked = int(len(non_zero_indices) * self.mask_ratio)
        # num_masked = 2
        num_masked = max(int(len(non_zero_indices) * self.mask_ratio), 2)
        masked_indices = np.random.choice(non_zero_indices.numpy(), num_masked, replace=False)

        # 创建 masked_indices0，初始全为 0
        masked_indices0 = torch.zeros_like(input_ids, dtype=torch.bool)

        # 将随机选择的位置标记为 1
        masked_indices0[masked_indices] = 1

        # 返回包含 input_ids, intensity, attention_mask 和 masked_indices0 的字典
        return {
            'input_ids': input_ids,
            'intensity': intensity,
            'attention_mask': attention_mask,
            'masked_indices0': masked_indices0
        }


def mask_input_ids(self, input_ids, p):
    mask_indices = []
    for token in input_ids:
        if token == 0:
            mask_indices.append(0)  # 元素为0的位置不参与mask
        else:
            mask_indices.append(1 if random.random() < p else 0)
    return mask_indices
import torch
from torch.utils.data import DataLoader

class DataSet:
    def __init__(self, max_len=100):
        # 超参数
        self.max_len = max_len

        # 加载数据
        train_ref, msms1, precursor1, smiles1 = ParseOrbitrap('D:/Projects/MSDiffAD/data/ob_train_ref.pickle')
        train_ref, word2idx = MakeTrainData(msms1, precursor1, self.max_len)

        # 词汇表大小
        self.vocab_size = len(word2idx)
        self.word2idx = word2idx

        # 准备数据集
        input_ids, intensity, _, _ = zip(*train_ref)
        intensity = [torch.FloatTensor(i) for i in intensity]
        input_ids = [torch.LongTensor(i) for i in input_ids]

        # 创建数据集
        self.dataset = MyDataSet(input_ids, intensity)

        # 生成数据加载器
        self.train_dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.valid_dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)  # 可以根据需要修改
        self.test_dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)  # 可以根据需要修改

        # 数据集的大小
        self.item_size = len(self.dataset)

    def get_dataloaders(self):
        return self.train_dataloader, self.valid_dataloader, self.test_dataloader




def main():
    args = get_config()
    set_seed(args.seed)
    max_len = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args_str = f"{args.model_name}-{args.dataset}-{args.model_idx}"
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    #训练数据
    data_generator = DataSet()

    trainer = Trainer(args, device, data_generator)
    epochs = 200
    train_time = []
    for epoch in range(epochs):
        t = trainer._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)
        train_time.append(t)


if __name__ == "__main__":
    main()






